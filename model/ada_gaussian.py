"""Continuous Gaussian feature field (paper equations 6--11).

Coordinates passed to query_field are (y, x) in [-1, 1]; internal centers
are (x, y). Template standard deviations are in encoder-grid pixel units.
The density is evaluated in normalized coordinates, independent of output size.
This exact PyTorch backend is intended as a correctness reference; it does not
claim the speed of a tiled CUDA splatter. Query chunks are checkpointed in
training to bound saved pairwise activations.
"""
import math
import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from utils import visualize_embeddings


def make_coord(shape, ranges=None, flatten=True):
    axes = []
    for i, n in enumerate(shape):
        if n <= 0:
            raise ValueError("Grid dimensions must be positive")
        lo, hi = (-1, 1) if ranges is None else ranges[i]
        axes.append(lo + (torch.arange(n).float() + 0.5) * (hi - lo) / n)
    grid = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)
    return grid.reshape(-1, len(shape)) if flatten else grid


def get_coord(width, height):
    """Row-major pixel centers in (x, y) order."""
    return make_coord((height, width)).flip(-1)


def nearest_sample(features, coord):
    """Nearest grid sampling with zero padding and align_corners=False.

    coord is [B,Q,2] in (y,x) order. Gather has a deterministic CUDA backward
    under torch.use_deterministic_algorithms(True), unlike grid_sample.
    """
    h, w = features.shape[-2:]
    y = torch.round(((coord[..., 0] + 1) * h - 1) / 2).long()
    x = torch.round(((coord[..., 1] + 1) * w - 1) / 2).long()
    valid = (y >= 0) & (y < h) & (x >= 0) & (x < w)
    index = y.clamp(0, h - 1) * w + x.clamp(0, w - 1)
    values = features.flatten(2).gather(2, index[:, None].expand(-1, features.shape[1], -1))
    return values.transpose(1, 2) * valid[..., None].to(features.dtype)


class AdaGsSplatting(nn.Module):
    def __init__(self, kernel_size, encoder_dim, hidden_dim=256,
                 num_points=500, pos_feat_dim=None, decoder_mode='frequency',
                 query_chunk_size=256):
        super().__init__()
        if decoder_mode not in ('field', 'frequency'):
            raise ValueError("decoder_mode must be field or frequency")
        if num_points <= 0 or query_chunk_size <= 0:
            raise ValueError("num_points and query_chunk_size must be positive")
        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim
        self.output_dim = encoder_dim if decoder_mode == 'field' else hidden_dim
        self.decoder_mode = decoder_mode
        self.num_points = num_points
        self.kernel_size = kernel_size  # Legacy CLI compatibility; no truncation.
        self.query_chunk_size = query_chunk_size
        # Raw, unconstrained parameters: positive scales, bounded correlation,
        # and sigmoid opacity are enforced on EVERY forward pass.
        sigma = torch.linspace(0.2, 3.0, num_points)
        self.raw_sigma_x = nn.Parameter(torch.log(torch.expm1(sigma - 1e-4)))
        self.raw_sigma_y = nn.Parameter(torch.log(torch.expm1(sigma - 1e-4)))
        self.raw_rho = nn.Parameter(torch.zeros(num_points))
        self.opacity_logits = nn.Parameter(torch.ones(num_points))
        self.mlp_offset = nn.Sequential(
            nn.Conv1d(pos_feat_dim or encoder_dim, hidden_dim, 1), nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 1), nn.ReLU(),
            nn.Conv1d(hidden_dim, 2, 1))
        nn.init.zeros_(self.mlp_offset[-1].weight)
        nn.init.zeros_(self.mlp_offset[-1].bias)
        if decoder_mode == 'frequency':
            if hidden_dim % 2:
                raise ValueError("Frequency decoder requires even hidden_dim")
            self.coef = nn.Conv2d(encoder_dim, hidden_dim, 3, padding=1)
            self.freq = nn.Conv2d(encoder_dim, hidden_dim, 3, padding=1)
            self.phase = nn.Linear(2, hidden_dim // 2, bias=False)

    def template_covariances(self):
        sx = F.softplus(self.raw_sigma_x) + 1e-4
        sy = F.softplus(self.raw_sigma_y) + 1e-4
        rho = torch.tanh(self.raw_rho) * (1 - 1e-4)
        xy = rho * sx * sy
        return torch.stack((sx.square(), xy, xy, sy.square()), -1).reshape(-1, 2, 2)

    def weighted_gaussian_parameters(self, weights):
        """Convex covariance mixture, then sigmoid of mixed opacity logits.

        weights are selection probabilities, NOT unnormalized logits.
        Returns covariance [B,N,2,2] and opacity [B,N].
        """
        if weights.ndim != 4 or weights.shape[1] != self.num_points:
            raise ValueError("Expected selection probabilities [B,num_points,H,W]")
        w = weights.flatten(2).transpose(1, 2)
        cov = torch.einsum('bnk,kij->bnij', w, self.template_covariances())
        opacity = torch.sigmoid(w @ self.opacity_logits)
        return cov, opacity

    def gaussian_parameters(self, feat, weights, pos_feat=None):
        b, c, h, w = feat.shape
        if c != self.encoder_dim or weights.shape[0] != b or weights.shape[-2:] != (h, w):
            raise ValueError("Features and selection probabilities must share their grid")
        pos = feat if pos_feat is None else pos_feat
        if pos.shape[0] != b or pos.shape[-2:] != (h, w):
            raise ValueError("Contour features must share the Gaussian grid")
        offsets = torch.tanh(self.mlp_offset(pos.flatten(2))).transpose(1, 2)
        centers = get_coord(w, h).to(feat).unsqueeze(0) + offsets
        cov, opacity = self.weighted_gaussian_parameters(weights)
        # S Sigma S^T converts LR-pixel covariance to normalized coordinates.
        units = feat.new_tensor([2 / w, 2 / h])
        cov = cov * units[:, None] * units[None, :]
        chol = torch.linalg.cholesky(cov)
        return centers, chol, opacity

    @staticmethod
    def _query_chunk(query_xy, centers, chol, values):
        delta = query_xy[:, :, None, :] - centers[:, None, :, :]
        # Analytic triangular solve avoids materializing a 2x2 inverse per query.
        dx = delta[..., 0] / chol[:, None, :, 0, 0]
        dy = (delta[..., 1] - chol[:, None, :, 1, 0] * dx) / chol[:, None, :, 1, 1]
        density = torch.exp(-0.5 * (dx.square() + dy.square()))
        determinant_sqrt = chol[..., 0, 0] * chol[..., 1, 1]
        density = density / (2 * math.pi * determinant_sqrt[:, None, :])
        return density @ values

    def query_field(self, feat, weights, coord, pos_feat=None):
        centers, chol, opacity = self.gaussian_parameters(feat, weights, pos_feat)
        if coord.ndim != 3 or coord.shape[0] != feat.shape[0] or coord.shape[-1] != 2:
            raise ValueError("Expected query coordinates [B,Q,2] in (y,x) order")
        values = feat.flatten(2).transpose(1, 2) * opacity.unsqueeze(-1)
        outputs = []
        for q in coord.flip(-1).split(self.query_chunk_size, dim=1):
            inputs = (q, centers, chol, values)
            if self.training and torch.is_grad_enabled():
                out = checkpoint(self._query_chunk, *inputs, use_reentrant=False)
            else:
                out = self._query_chunk(*inputs)
            outputs.append(out)
        return torch.cat(outputs, dim=1)

    def splatting_features(self, feat, logits, scale, pos_feat=None):
        if isinstance(scale, (tuple, list, torch.Size)):
            h, w = map(int, scale)
        else:
            h, w = (round(feat.shape[-2] * float(scale)), round(feat.shape[-1] * float(scale)))
        coord = make_coord((h, w)).to(feat).unsqueeze(0).expand(feat.shape[0], -1, -1)
        field = self.query_field(feat, logits, coord, pos_feat)
        return field.transpose(1, 2).reshape(feat.shape[0], self.encoder_dim, h, w)

    def augmentation(self,feat,final_feats,coord,cell=None):
         # 8.1 特征提取和初始化
        coef = self.coef(final_feats)
        freq = self.freq(final_feats)
        feat_coord = self.feat_coord
        q_coef = nearest_sample(coef, coord)
        q_freq = nearest_sample(freq, coord)
        q_coord = nearest_sample(feat_coord, coord)
        # 8.3 计算相对坐标
        rel_coord = coord - q_coord
        rel_coord[:, :, 0] *= feat.shape[-2]  # 高度方向缩放
        rel_coord[:, :, 1] *= feat.shape[-1]  # 宽度方向缩放
        # 单元格的处理
        rel_cell = cell.clone()
        rel_cell[:, :, 0] *= feat.shape[-2]
        rel_cell[:, :, 1] *= feat.shape[-1]
        bs, q = coord.shape[:2]
        # 8.4 计算频率
        q_freq = torch.stack(torch.split(q_freq, 2, dim=-1), dim=-1)
        q_freq = torch.mul(q_freq, rel_coord.unsqueeze(-1))
        q_freq = torch.sum(q_freq, dim=-2)
        # 8.5 频率相位调整和三角函数变换
        q_freq += self.phase(rel_cell.view((bs * q, -1))).view(bs, q, -1) # 使用相位变换网络调整
        q_freq = torch.cat((torch.cos(np.pi * q_freq), torch.sin(np.pi * q_freq)), dim=-1)
        # 8.6 计算最终的输入
        rendered_feat = torch.mul(q_coef, q_freq)
        return rendered_feat

    def forward(self, inp, feat, logits, scale, coord, cell=None, save_path=None, pos_feat=None):
        if self.decoder_mode == 'field':
            return self.query_field(feat, logits, coord, pos_feat)
        final_feats = self.splatting_features(feat, logits, scale, pos_feat)
        self.feat_coord = make_coord(final_feats.shape[-2:], flatten=False).to(feat).permute(2, 0, 1).unsqueeze(0).expand(feat.shape[0], -1, -1, -1)
        if save_path:
            visualize_embeddings(final_feats, os.path.join(save_path, 'gs_features'))
        # Training enforces label_size == image_size; evaluation restores image_size
        # from the checkpoint. Keep that phase condition when the output grid changes.
        # The Gaussian field above is still queried directly at the target resolution.
        phase_cell = torch.ones_like(coord)
        phase_cell[..., 0] *= 2 / inp.shape[-2]
        phase_cell[..., 1] *= 2 / inp.shape[-1]
        return self.augmentation(feat, final_feats, coord, phase_cell)

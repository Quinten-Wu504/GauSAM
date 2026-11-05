"""加入lora模块，对整个SAM相关的进行了一次封装"""

import torch
from torch import nn
from torch.nn import functional as F
import math
from typing import Any, Dict, List, Tuple

from .segment_anything.modeling.sam import Sam
from .segment_anything.modeling.common import LayerNorm2d
class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)

class FeatureFusionConv(nn.Module):
    """高维特征融合网络，将256维度的图像特征降维到指定维度"""
    def __init__(self, input_channels=256, output_channels=64):
        super(FeatureFusionConv, self).__init__()
        # output_channels=min(output_channels, input_channels)
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x 的形状：(batch_size, input_channels, H, W)
        x = x.contiguous()
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# from I-MedSAM LORA###
class LoRA_qkv(nn.Module):
    def __init__(
            self,
            qkv: nn.Module,
            q_lora_a: nn.Module,
            q_lora_b: nn.Module,
            k_lora_a: nn.Module,
            k_lora_b: nn.Module,
            v_lora_a: nn.Module,
            v_lora_b: nn.Module,
            w_gate: nn.Module,
            mode: str = "qv",
    ):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.w_gate = w_gate
        self.mode = mode
        self.q_lora_a = q_lora_a
        self.q_lora_b = q_lora_b
        self.k_lora_a = k_lora_a
        self.k_lora_b = k_lora_b
        self.v_lora_a = v_lora_a
        self.v_lora_b = v_lora_b

    def forward(self, x):
        qkv = self.qkv(x)

        f = x.mean(dim=1).mean(dim=1)
        g = self.w_gate(f)

        if 'q' in self.mode:
            q_lora = [self.q_lora_b[0](self.q_lora_a[0](x))]
            q_lora = torch.stack(q_lora, dim=1)
            q_lora = torch.einsum('bemnc,be->bemnc', q_lora, g).sum(dim=1)
            qkv[:, :, :, :self.dim] += q_lora
        if 'k' in self.mode:
            k_lora = [self.k_lora_b[0](self.k_lora_a[0](x))]
            k_lora = torch.stack(k_lora, dim=1)
            k_lora = torch.einsum('bemnc,be->bemnc', k_lora, g).sum(dim=1)
            qkv[:, :, :, self.dim:2*self.dim] += k_lora
        if 'v' in self.mode:
            v_lora = [self.v_lora_b[0](self.v_lora_a[0](x))]
            v_lora = torch.stack(v_lora, dim=1)
            v_lora = torch.einsum('bemnc,be->bemnc', v_lora, g).sum(dim=1)
            qkv[:, :, :, -self.dim:] += v_lora

        return qkv

# 设计一个新的图像编码器，提取SAM中TwoWayTransformer输出后的融合了prompt和图像特征的特征
class SamEncoder(nn.Module):
    def __init__(
            self,
            sam: Sam,
            r: int = 4,
            mode: str ='qv',
            pos: list = None,
            n_class: int = 500,
            n_feat: int = 64,
            no_upsampling: bool = True#,
            # pos_feat_dim: int = 768
            ) -> None:
        super().__init__()
        self.sam = sam
        self.no_upsampling = no_upsampling
        self.feature_dim = 256 if no_upsampling else 64
        self.feature_fusion = (FeatureFusionConv(self.feature_dim, n_feat)
                               if self.feature_dim != n_feat else nn.Identity())
        self.contour_fusion = (nn.Conv2d(self.feature_dim, n_feat, 1)
                               if self.feature_dim != n_feat else nn.Identity())
        # n_feat = min(self.feature_dim,n_feat)
        self.segmentation_head = SegmentationHead(in_channels= n_feat, out_channels=n_class)
        # self.custom_upscaling = nn.Sequential(
        #     nn.ConvTranspose2d(pos_feat_dim, pos_feat_dim // 2, kernel_size=2, stride=2),
        #     LayerNorm2d(pos_feat_dim // 2),
        #     nn.GELU(),
        #     nn.ConvTranspose2d(pos_feat_dim // 2, pos_feat_dim // 4, kernel_size=2, stride=2),
        #     nn.GELU(),
        # )
        if pos:
            self.pos = pos
        else:
            self.pos = list(range(len(self.sam.image_encoder.blocks)))

        for name, param in sam.image_encoder.named_parameters():
            if ("contour" not in name):
                param.requires_grad = False
            if ("pos_patch_embed" in name):
                param.requires_grad = True
        for param in self.sam.prompt_encoder.parameters():
            param.requires_grad = False
        for name, param in sam.mask_decoder.named_parameters():
            if "custom_upscaling" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.w_As = []
        self.w_Bs = []
        self.w_Gs = []
        # apply LoRA to SAM image encoder
        for idx, blk in enumerate(self.sam.image_encoder.blocks):
            if idx not in self.pos:
                continue

            qkv = blk.attn.qkv
            self.dim = qkv.in_features

            if 'q' in mode:
                q_lora_a = [nn.Linear(self.dim, r, bias=False)]
                q_lora_b = [nn.Linear(r, self.dim, bias=False)]
                self.w_As += q_lora_a
                self.w_Bs += q_lora_b
            else:
                q_lora_a = None
                q_lora_b = None
            if 'k' in mode:
                k_lora_a = [nn.Linear(self.dim, r, bias=False)]
                k_lora_b = [nn.Linear(r, self.dim, bias=False)]
                self.w_As += k_lora_a
                self.w_Bs += k_lora_b
            else:
                k_lora_a = None
                k_lora_b = None
            if 'v' in mode:
                v_lora_a = [nn.Linear(self.dim, r, bias=False)]
                v_lora_b = [nn.Linear(r, self.dim, bias=False)]
                self.w_As += v_lora_a
                self.w_Bs += v_lora_b
            else:
                v_lora_a = None
                v_lora_b = None

            gate = nn.Linear(self.dim, 1, bias=False)
            self.w_Gs.append(gate)

            blk.attn.qkv = LoRA_qkv(
                qkv,
                nn.ModuleList(q_lora_a),
                nn.ModuleList(q_lora_b),
                nn.ModuleList(k_lora_a),
                nn.ModuleList(k_lora_b),
                nn.ModuleList(v_lora_a),
                nn.ModuleList(v_lora_b),
                gate,
                mode=mode,
            )

        # init LoRA layer parameters
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self,
                img: torch,
                points: torch = None,
                boxes: torch = None,
                masks:torch = None,
                ) -> torch.Tensor:
        # Datasets return float RGB in [0,1]; SAM expects [0,255].
        input_img = self.sam.preprocess(img * 255.0)
        image_embeddings, pfeat_embeddings = self.sam.image_encoder(input_img)#,         pfeat_embeddings = None
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=masks
        )
        features = self.sam.mask_decoder.get_embeddings(
            image_embeddings=image_embeddings, # [B,256,64,64]
            image_pe=self.sam.prompt_encoder.get_dense_pe(), # [1,256,64,64]
            sparse_prompt_embeddings=sparse_embeddings, # [B,2,256]
            dense_prompt_embeddings=dense_embeddings, # [B,256,64,64]
            no_upsampling=self.no_upsampling
        )
        pfeat_embeddings = self.sam.mask_decoder.get_embeddings(
            image_embeddings = pfeat_embeddings, # [B,256,64,64]
            image_pe=self.sam.prompt_encoder.get_dense_pe(), # [1,256,64,64]
            sparse_prompt_embeddings=sparse_embeddings, # [B,2,256]
            dense_prompt_embeddings=dense_embeddings, # [B,256,64,64]
            no_upsampling=self.no_upsampling
        )
        # pfeat_embeddings = self.custom_upscaling(pfeat_embeddings)
        # feature = self.sam(img, multimask_output=False, no_upsampling=self.no_upsampling)
        features = self.feature_fusion(features)
        pfeat_embeddings = self.contour_fusion(pfeat_embeddings)
        logits = self.segmentation_head(features)
        B, Class, H, W = logits.shape
        logits = logits.permute(0, 2, 3, 1).contiguous().view(B * H * W, Class)
        if self.training:
            logits = F.gumbel_softmax(logits, tau=1, hard=False)
        if not self.training:
            logits = F.one_hot(logits.argmax(dim=-1), num_classes=Class).to(logits.dtype)
        logits = logits.view(B, H, W, Class).permute(0, 3, 1, 2).contiguous()
        return features, logits, pfeat_embeddings

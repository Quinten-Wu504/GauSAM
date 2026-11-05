"""Foreground segmentation from SAM features and a continuous Gaussian field."""
import torch
import torch.nn as nn
from .ada_gaussian import make_coord


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list, act='gelu'):
        super().__init__()
        if act is None:
            self.act = None
        elif act.lower() == 'relu':
            self.act = nn.ReLU()
        elif act.lower() == 'gelu':
            self.act = nn.GELU()
        else:
            assert False, f'activation {act} is not supported'
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            if self.act:
                layers.append(self.act)
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)

class GauSAM(nn.Module):
    """SAM encoder and Gaussian field with one foreground-logit decoder."""
    def __init__(self, sam_encoder, gaussian_model, hidden_list=(256, 256)):
        super().__init__()
        self.sam_encoder = sam_encoder
        self.gaussian_model = gaussian_model
        # Retain dec.0 keys so existing foreground checkpoints still load.
        self.dec = nn.ModuleList([MLP(gaussian_model.output_dim, 1, hidden_list)])

    def forward(self, input, boxes, tar_size, save_path=None):
        feature, logits, pfeat = self.sam_encoder(input, boxes=boxes)
        tar_size = tuple(tar_size) if tar_size else tuple(input.shape[-2:])
        if len(tar_size) != 2 or min(tar_size) <= 0:
            raise ValueError("tar_size must contain positive height and width")
        h, w = tar_size
        coord = make_coord(tar_size).to(feature).unsqueeze(0).expand(feature.shape[0], -1, -1)
        cell = torch.ones_like(coord)
        cell[..., 0] *= 2 / h
        cell[..., 1] *= 2 / w
        features = self.gaussian_model(input, feature, logits, tar_size, coord, cell,
                                      save_path, pos_feat=pfeat)
        pred = self.dec[0](features.contiguous())
        return pred.view(feature.shape[0], h, w, 1).permute(0, 3, 1, 2).contiguous()

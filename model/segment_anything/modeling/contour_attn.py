import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Type

from .common import MLPBlock
from .spatial_attn import SpatialAttention
from .non_subsampled_contourlet import non_subsampled_contourlet_decompose


class ContourExtractor:
    def __init__(self, levels=(3, 3, 3), pfilter_name='pyr', dfilter_name='pkva'):
        """
        NSCT边缘图提取器

        Args:
            levels: list[int] 每层的方向数，如 [2,2,3]
            pfilter_name: 金字塔滤波器名，如 'pyr'
            dfilter_name: 方向滤波器名，如 'pkva'
            device: 使用的设备，如 'cuda' 或 'cpu'
        """
        self.levels = levels
        self.pfilter_name = pfilter_name
        self.dfilter_name = dfilter_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def extract(self, images):
        """
        提取 NSCT 边缘图。

        Args:
            images: [B, C, H, W] 张量
        Returns:
            edge_map: [B, C, H, W] 边缘响应图，范围在 [0, 1]
        """
        # Follow the input device, including the local DDP rank.
        device = images.device

        _, contourlet_coeffs = non_subsampled_contourlet_decompose(
            images, levels=self.levels,
            pfilter_name=self.pfilter_name,
            dfilter_name=self.dfilter_name, device=device,
        )

        edge_map = torch.zeros_like(images)
        for coeff in contourlet_coeffs[1:]:
            if isinstance(coeff, list):
                for subband in coeff:
                    edge_map = edge_map + torch.log1p(torch.abs(subband)).pow(0.7)
            else:
                edge_map = edge_map + torch.log1p(torch.abs(coeff)).pow(0.7)

        # Paper Eq. (4): nonlinear compression precedes subband summation.

        # 标准化
        edge_map -= edge_map.amin(dim=(-2, -1), keepdim=True)
        edge_map /= (edge_map.amax(dim=(-2, -1), keepdim=True) + 1e-8)

        return torch.clamp(edge_map, 0, 1)


class ContourAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        act_layer: Type[nn.Module] = nn.GELU,
    ):
        super(ContourAttention, self).__init__()
        self.lin1 = nn.Linear(dim, 4)
        self.lin2 = nn.Linear(4, dim)
        self.act = act_layer()

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        x = self.lin1(x)
        x = self.act(x)
        x = self.lin2(x)

        return x


class ContourBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        use_contourlet: bool = True,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super(ContourBlock, self).__init__()
        self.use_contourlet = use_contourlet
        # 对特征先进行层归一化
        self.norm1 = norm_layer(dim)

        # 位置编码信息的注意力
        self.attn = SpatialAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        if self.use_contourlet:
            self.norm_contour = norm_layer(dim)
            self.contour_attn = ContourAttention(dim)

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(
        self,
        x: torch.Tensor,
        x_contour: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        if self.use_contourlet:
            x_contour_norm = self.norm_contour(x_contour)
            x_contour_weight = self.contour_attn(x_contour_norm)
            x_contour = x_contour_weight * x_contour
            x = x + self.mlp(self.norm2(x)) + 0.5 * x_contour
            # print('进入了contour')
        else:
            x_contour = None
            x = x + self.mlp(self.norm2(x))

        return x, x_contour


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x

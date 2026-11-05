import math
import torch
import torch.nn.functional as F


# 预定义滤波器矩阵为全局常量
H0 = torch.tensor([
    [-0.003236043456039806, -0.012944173824159223, -0.019416260736238835],
    [-0.012944173824159223, 0.0625, 0.15088834764831843],
    [-0.019416260736238835, 0.15088834764831843, 0.3406092167691145]
], dtype=torch.float32)

G1 = torch.tensor([
    [-0.003236043456039806, -0.012944173824159223, -0.019416260736238835],
    [-0.012944173824159223, -0.0625, -0.09911165235168155],
    [-0.019416260736238835, -0.09911165235168155, 0.8406092167691145]
], dtype=torch.float32)

G0 = torch.tensor([
    [-0.00016755163599004882, -0.001005309815940293, -0.002513274539850732, -0.003351032719800976],
    [-0.001005309815940293, -0.005246663087920392, -0.01193886400821893, -0.015395021472477663],
    [-0.002513274539850732, -0.01193886400821893, 0.06769410071569153, 0.15423938036811946],
    [-0.003351032719800976, -0.015395021472477663, 0.15423938036811946, 0.3325667382415921]
], dtype=torch.float32)

H1 = torch.tensor([
    [0.00016755163599004882, 0.001005309815940293, 0.002513274539850732, 0.003351032719800976],
    [0.001005309815940293, -0.0012254238241592198, -0.013949483640099517, -0.023437500000000007],
    [0.002513274539850732, -0.013949483640099517, -0.06769410071569153, -0.10246268507148255],
    [0.003351032719800976, -0.023437500000000007, -0.10246268507148255, 0.8486516952966369]
], dtype=torch.float32)

V = torch.tensor([
    [0.6300, -0.1930, 0.0972, -0.0526, 0.0272, -0.0144]
], dtype=torch.float32)


def resamp(x, resamp_type, shift=1, device="cuda" if torch.cuda.is_available() else "cpu"):
    # 将输入张量移到指定设备
    x = torch.tensor(x, dtype=torch.float32, device=device) if not isinstance(x, torch.Tensor) else x.to(device)

    H, W = x.shape
    if resamp_type == 3 or resamp_type == 4:
        # 初始化输出张量
        y = torch.zeros((H, W + abs(shift * (H - 1))), dtype=torch.float32, device=device)

        # 计算移位索引
        offset = torch.arange(H, device=device)
        shift_idx = offset * (-shift if resamp_type == 3 else shift)
        if shift_idx[-1] < 0:
            shift_idx = shift_idx - shift_idx[-1]
        shift_idx = shift_idx.long()

        # 向量化填充数据
        rows = offset.repeat_interleave(W)
        cols = (shift_idx.repeat_interleave(W) + torch.arange(W, device=device).repeat(H)).long()
        y[rows, cols] = torch.flatten(x)

        # 修剪零列
        start = 0
        u, s, v = torch.linalg.svd(y[:, start].reshape(1, -1), full_matrices=False)
        while s.max() == 0:
            start += 1
            u, s, v = torch.linalg.svd(y[:, start].reshape(1, -1), full_matrices=False)

        finish = y.shape[1] - 1
        u, s, v = torch.linalg.svd(y[:, finish].reshape(1, -1), full_matrices=False)
        while s.max() == 0:
            finish -= 1
            u, s, v = torch.linalg.svd(y[:, finish].reshape(1, -1), full_matrices=False)

        y = y[:, start:finish + 1]
    elif resamp_type == 1 or resamp_type == 2:
        # 初始化输出张量
        y = torch.zeros((H + abs(shift * (W - 1)), W), dtype=torch.float32, device=device)

        # 计算移位索引
        shift_idx = torch.arange(W, device=device) * (-shift if resamp_type == 1 else shift)
        if shift_idx[-1] < 0:
            shift_idx = shift_idx - shift_idx[-1]
        shift_idx = shift_idx.long()

        # 向量化填充数据
        cols = torch.arange(W, device=device).repeat_interleave(H)
        rows = (shift_idx.repeat_interleave(H) + torch.arange(H, device=device).repeat(W)).long()
        y[rows, cols] = torch.flatten(x.t())

        # 修剪零行
        start = 0
        u, s, v = torch.linalg.svd(y[start, :].reshape(1, -1), full_matrices=False)
        while s.max() == 0:
            start += 1
            u, s, v = torch.linalg.svd(y[start, :].reshape(1, -1), full_matrices=False)

        finish = y.shape[0] - 1
        u, s, v = torch.linalg.svd(y[finish, :].reshape(1, -1), full_matrices=False)
        while s.max() == 0:
            finish -= 1
            u, s, v = torch.linalg.svd(y[finish, :].reshape(1, -1), full_matrices=False)

        y = y[start:finish + 1]
    else:
        raise ValueError('func resamp: Invalid resamp_type! ')

    return y


def qup(x, up_type, device="cuda" if torch.cuda.is_available() else "cpu"):
    # 将输入张量移到指定设备
    x = torch.tensor(x, dtype=torch.float32, device=device) if not isinstance(x, torch.Tensor) else x.to(device)

    if up_type == 1:
        x1 = resamp(x, resamp_type=4, shift=1, device=device)
        H, W = x1.shape

        # 初始化中间张量
        x2 = torch.zeros((2 * H - 1, W), dtype=torch.float32, device=device)

        # 向量化填充偶数行
        even_rows = torch.arange(0, 2 * H - 1, 2, device=device)
        x2[even_rows] = x1

        # 第二次重采样
        y = resamp(x2, resamp_type=1, shift=1, device=device)
    else:
        raise ValueError("func qup: Invalid up_type!")

    return y


def repeat_matrix(A, M, N, device="cuda" if torch.cuda.is_available() else "cpu"):
    # 将输入张量移到指定设备
    A = torch.tensor(A, dtype=torch.float32, device=device) if not isinstance(A, torch.Tensor) else A.to(device)

    # 处理 M, N 输入
    if isinstance(N, torch.Tensor):
        N = N.tolist()  # 转换为列表以检查是否为空
    if isinstance(M, torch.Tensor):
        M = M.tolist()  # 转换为列表以提取 siz

    if N is None:
        if isinstance(M, (list, tuple)) and len(M) > 1:
            siz = M
        else:
            raise ValueError("func repeat_matrix: Invalid M when N is empty!")
    else:
        raise ValueError("func repeat_matrix: Non-empty N is not supported!")

    # 验证 siz 为 2 元素列表
    if len(siz) != 2:
        raise ValueError("func repeat_matrix: siz must be a 2-element list!")

    H, W = A.shape
    if H == 1 and siz[1] == 1:
        # A 是行向量，siz[1]=1，生成 B = ones(siz[0], 1) * A
        ones = torch.ones((siz[0], 1), dtype=torch.float32, device=device)
        B = torch.outer(ones.squeeze(1), A.squeeze(0))
    elif W == 1 and siz[0] == 1:
        # A 是列向量，siz[0]=1，生成 B = A * ones(1, siz[1])
        ones = torch.ones((1, siz[1]), dtype=torch.float32, device=device)
        B = torch.outer(A.squeeze(1), ones.squeeze(0))
    else:
        raise ValueError("func repeat_matrix: Invalid A shape or siz!")

    return B


def modulate2(x, mod_type, center=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    # 将输入张量移到指定设备
    x = torch.tensor(x, dtype=torch.float32, device=device) if not isinstance(x, torch.Tensor) else x.to(device)

    # 默认中心点
    center = [0, 0] if center is None else center
    center = torch.tensor(center, dtype=torch.float32, device=device)

    # 获取输入形状
    H, W = x.shape

    # 计算偏移索引
    o = torch.tensor([int(H / 2.0) + 1 + center[0], int(W / 2.0) + 1 + center[1]], device=device).long()
    n1 = torch.arange(1, H + 1, device=device) - o[0]
    n2 = torch.arange(1, W + 1, device=device) - o[1]

    if mod_type == 'c':
        # 列调制
        m2 = torch.pow(-1, n2.abs()).reshape(1, -1)
        M = [H, 1]
        y = x * repeat_matrix(m2, M, None, device=device)
    elif mod_type == 'r':
        # 行调制
        m1 = torch.pow(-1, n1.abs()).reshape(-1, 1)
        M = [1, W]
        y = x * repeat_matrix(m1, M, None, device=device)
    elif mod_type == 'b':
        # 行和列调制
        m1 = torch.pow(-1, n1.abs()).reshape(-1, 1)
        m2 = torch.pow(-1, n2.abs()).reshape(1, -1)
        m = torch.outer(m1.squeeze(1), m2.squeeze(0))
        y = x * m
    else:
        raise ValueError("func modulate2: Invalid mod_type! Must be 'c', 'r', or 'b'!")

    return y


def ld2quin(beta, device="cuda" if torch.cuda.is_available() else "cpu"):
    # 将输入张量移到指定设备
    beta = torch.tensor(beta, dtype=torch.float32, device=device) if not isinstance(beta, torch.Tensor) else beta.to(device)

    # 验证输入为 1D 滤波器
    if beta.shape[0] != 1:
        raise ValueError("func ld2quin: Input must be a 1-D filter!")

    lf = beta.shape[1]
    n = lf // 2

    # 计算外积
    sp = torch.outer(torch.flatten(beta), torch.flatten(beta))
    # 调用 qup
    h = qup(sp, up_type=1, device=device)

    # 计算 h0
    h0 = h.clone()  # 克隆以避免修改 h
    h0[2 * n - 1, 2 * n - 1] += 1
    h0 = h0 / 2

    # 计算 h1（2D 卷积）
    h0_4d = torch.flip(h0, dims=[0, 1]).unsqueeze(0).unsqueeze(0)  # 形状 [1, 1, W, H]
    h_4d = h.unsqueeze(0).unsqueeze(0)  # 形状 [1, 1, H, W]
    # 设置 padding 以模拟 'full' 卷积
    padding = h0.shape[0] - 1
    h1 = -F.conv2d(h_4d, h0_4d, padding=padding).squeeze(0).squeeze(0)  # 移除批次和通道维度
    h1[4 * n - 2, 4 * n - 2] += 1

    return [h0, h1]


def ldfilter(filter_name, device="cuda" if torch.cuda.is_available() else "cpu"):
    if filter_name == 'pkva':
        # 将常量移动到指定设备
        v = V.to(device)
        # 水平翻转
        v_flipped = torch.fliplr(v)
        # 拼接翻转后的矩阵和原矩阵
        f = torch.cat((v_flipped, v), dim=1)
        return f

    return None


def dfilters(filter_name, filter_type, device="cuda" if torch.cuda.is_available() else "cpu"):
    if filter_name == 'pkva':
        # 获取 1D 滤波器
        beta = ldfilter(filter_name, device=device)

        # 计算 2D 滤波器
        h0, h1 = ld2quin(beta, device=device)

        # 缩放滤波器
        scale = torch.tensor(2.0, device=device).sqrt()
        h0 = h0 * scale
        h1 = h1 * scale

        if filter_type == 'r':
            # 假设 modulate2 接受 PyTorch 张量并返回张量
            f0 = modulate2(h1, 'b', None, device=device)
            f1 = modulate2(h0, 'b', None, device=device)
            h0 = f0
            h1 = f1
    else:
        raise ValueError("func dfilters: Invalid filter_name!")

    return [h0, h1]


def pfilters(f1, f2, device="cuda" if torch.cuda.is_available() else "cpu"):
    # 将输入张量移到指定设备
    f1 = torch.tensor(f1, dtype=torch.float32, device=device) if not isinstance(f1, torch.Tensor) else f1.to(device)
    f2 = torch.tensor(f2, dtype=torch.float32, device=device) if not isinstance(f2, torch.Tensor) else f2.to(device)

    # 验证输入形状
    if f1.ndim != 2 or f2.ndim != 2:
        raise ValueError("func pfilters: f1 and f2 must be 2D tensors!")

    # 初始化滤波器列表
    y1 = [None] * 4
    y2 = [None] * 4

    # 应用 modulate2
    y1[0] = modulate2(f1, 'r', None, device=device)
    y1[1] = modulate2(f1, 'c', None, device=device)
    y2[0] = modulate2(f2, 'r', None, device=device)
    y2[1] = modulate2(f2, 'c', None, device=device)
    y1[2] = y1[0].T
    y1[3] = y1[1].T
    y2[2] = y2[0].T
    y2[3] = y2[1].T

    # 批量应用 resamp
    y1 = [resamp(y1[i], i + 1, shift=1, device=device) for i in range(4)]
    y2 = [resamp(y2[i], i + 1, shift=1, device=device) for i in range(4)]

    return [y1, y2]


def atrous_filters(filter_name, device="cuda" if torch.cuda.is_available() else "cpu"):
    if filter_name == 'pyr':
        # 将常量移动到指定设备
        filters = [
            H0.to(device),
            H1.to(device),
            G0.to(device),
            G1.to(device)
        ]

        # 统一处理翻转和拼接
        for i, f in enumerate(filters):
            # 水平翻转并拼接（去掉最后一列以避免重复）
            f_lr = torch.fliplr(f[:, :-1])
            f = torch.cat((f, f_lr), dim=1)
            # 垂直翻转并拼接（去掉最后一行以避免重复）
            f_ud = torch.flipud(f[:-1, :])
            filters[i] = torch.cat((f, f_ud), dim=0)

        return filters

    return None


def zconv_transform(images, f, M, device="cuda" if torch.cuda.is_available() else "cpu"):
    # 将输入张量移到指定设备
    images = torch.tensor(images, dtype=torch.float32, device=device) if not isinstance(images, torch.Tensor) else images.to(device)
    f = torch.tensor(f, dtype=torch.float32, device=device) if not isinstance(f, torch.Tensor) else f.to(device)
    M = torch.tensor(M, dtype=torch.float32, device=device) if not isinstance(M, torch.Tensor) else M.to(device)

    # 验证输入
    if images.ndim != 4 or f.ndim != 2 or M.shape != (2, 2):
        raise ValueError("func zconv_transform: images must be 4D [B, C, H, W], f must be 2D, M must be 2x2!")

    # 获取批次大小和通道数
    B, C, H, W = images.shape

    # 计算索引范围
    H_f, W_f = f.shape
    terms_r = torch.tensor([
        1,
        1 - (H_f - 1) * M[0, 0],
        1 - (W_f - 1) * M[1, 0],
        1 - (H_f - 1) * M[0, 0] - (W_f - 1) * M[1, 0]
    ], device=device)
    terms_c = torch.tensor([
        1,
        1 - (H_f - 1) * M[0, 1],
        1 - (W_f - 1) * M[1, 1],
        1 - (H_f - 1) * M[0, 1] - (W_f - 1) * M[1, 1]
    ], device=device)

    new_f_start_r = torch.min(terms_r)
    new_f_end_r = torch.max(terms_r)
    new_f_start_c = torch.min(terms_c)
    new_f_end_c = torch.max(terms_c)
    new_f_r = (new_f_end_r - new_f_start_r + 1).floor().long()
    new_f_c = (new_f_end_c - new_f_start_c + 1).floor().long()

    # 创建新滤波器
    new_f = torch.zeros((new_f_r, new_f_c), dtype=torch.float32, device=device)

    # 生成索引
    i, j = torch.meshgrid(torch.arange(H_f, device=device), torch.arange(W_f, device=device))
    i, j = torch.flatten(i), torch.flatten(j)
    row_idx = (2 - i * M[0, 0] - j * M[1, 0] - new_f_start_r - 1).floor().long() % new_f_r
    col_idx = (2 - i * M[0, 1] - j * M[1, 1] - new_f_start_c - 1).floor().long() % new_f_c

    # 向量化填充 new_f
    new_f[row_idx.contiguous().view(-1), col_idx.contiguous().view(-1)] = f.contiguous().view(-1)

    # 调整滤波器形状以支持多通道卷积
    # 形状 [C, 1, H_f, W_f]：C 个输出通道，每个通道使用相同的单通道滤波器
    new_f = new_f.unsqueeze(0).unsqueeze(0).expand(C, 1, new_f_r, new_f_c)

    # 翻转卷积核以匹配 ndimage.convolve
    new_f = torch.flip(new_f, dims=[2, 3])

    # 计算填充大小
    pad_h = (new_f.shape[2] - 1) // 2
    pad_w = (new_f.shape[3] - 1) // 2

    # 应用周期性填充
    images_padded = F.pad(images, (pad_w, pad_w, pad_h, pad_h), mode='circular')

    # 应用卷积，groups=C 确保每个输入通道独立卷积
    images_out = F.conv2d(images_padded, new_f, groups=C)

    return images_out


def non_subsampled_pyramid_decompose(images, h0, h1, level, device="cuda" if torch.cuda.is_available() else "cpu"):
    # 将输入张量移到指定设备
    images = torch.tensor(images, dtype=torch.float32, device=device) if not isinstance(images, torch.Tensor) else images.to(device)
    h0 = torch.tensor(h0, dtype=torch.float32, device=device) if not isinstance(h0, torch.Tensor) else h0.to(device)
    h1 = torch.tensor(h1, dtype=torch.float32, device=device) if not isinstance(h1, torch.Tensor) else h1.to(device)

    # 验证输入
    if images.ndim != 4 or h0.ndim != 2 or h1.ndim != 2:
        raise ValueError("func non_subsampled_pyramid_decompose: images must be 4D [B, C, H, W], h0, h1 must be 2D!")
    if not isinstance(level, int) or level < 0:
        raise ValueError("func non_subsampled_pyramid_decompose: level must be a non-negative integer!")

    # 获取批次大小和通道数
    B, C, H, W = images.shape

    # 处理 h0
    H_f, W_f = h0.shape
    scale = 2 ** level
    new_h0_shape = (scale * H_f, scale * W_f)
    new_h0 = torch.zeros(new_h0_shape, dtype=torch.float32, device=device)

    # 生成索引（步长为 2^level）
    rows = torch.arange(0, new_h0_shape[0], scale, device=device)
    cols = torch.arange(0, new_h0_shape[1], scale, device=device)
    row_idx, col_idx = torch.meshgrid(rows, cols)
    indices = torch.stack([row_idx.contiguous().view(-1), col_idx.contiguous().view(-1)], dim=1)

    # 向量化填充 h0
    h0_flat = h0.view(-1)
    if indices.shape[0] != h0_flat.shape[0]:
        raise ValueError("func nsp_decompose: h0 size does not match index count!")
    new_h0[indices[:, 0], indices[:, 1]] = h0_flat

    # 裁剪 new_h0
    new_h0 = new_h0[:(H_f - 1) * scale + 1, :(W_f - 1) * scale + 1]

    # 处理 h1
    H_f, W_f = h1.shape
    new_h1_shape = (scale * H_f, scale * W_f)
    new_h1 = torch.zeros(new_h1_shape, dtype=torch.float32, device=device)

    # 重用索引（假设 h0 和 h1 形状相同）
    if h1.shape != h0.shape:
        rows = torch.arange(0, new_h1_shape[0], scale, device=device)
        cols = torch.arange(0, new_h1_shape[1], scale, device=device)
        row_idx, col_idx = torch.meshgrid(rows, cols)
        indices = torch.stack([row_idx.contiguous().view(-1), col_idx.contiguous().view(-1)], dim=1)

    # 向量化填充 h1
    h1_flat = h1.view(-1)
    if indices.shape[0] != h1_flat.shape[0]:
        raise ValueError("func nsp_decompose: h1 size does not match index count!")
    new_h1[indices[:, 0], indices[:, 1]] = h1_flat

    # 裁剪 new_h1
    new_h1 = new_h1[:(H_f - 1) * scale + 1, :(W_f - 1) * scale + 1]

    # 调整滤波器形状以支持多通道卷积
    # 形状 [C, 1, H_f, W_f]：C 个输出通道，每个通道使用相同的单通道滤波器
    new_h0_H, new_h0_W = new_h0.shape
    new_h1_H, new_h1_W = new_h1.shape
    new_h0 = new_h0.unsqueeze(0).unsqueeze(0).expand(C, 1, new_h0_H, new_h0_W)
    new_h1 = new_h1.unsqueeze(0).unsqueeze(0).expand(C, 1, new_h1_H, new_h1_W)

    # 翻转卷积核以匹配 cv2.filter2D
    new_h0 = torch.flip(new_h0, dims=[2, 3])
    new_h1 = torch.flip(new_h1, dims=[2, 3])

    # 计算填充大小
    pad_h = (new_h0.shape[2] - 1) // 2
    pad_w = (new_h0.shape[3] - 1) // 2
    # 应用反射填充
    images_padded_h0 = F.pad(images, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
    # 应用卷积，groups=C 确保每个输入通道独立卷积
    images_low = F.conv2d(images_padded_h0, new_h0, groups=C)  # 形状 [B, C, H, W]

    # 计算填充大小
    pad_h = (new_h1.shape[2] - 1) // 2
    pad_w = (new_h1.shape[3] - 1) // 2
    # 应用反射填充
    images_padded_h1 = F.pad(images, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
    # 应用卷积，groups=C 确保每个输入通道独立卷积
    images_high = F.conv2d(images_padded_h1, new_h1, groups=C)  # 形状 [B, C, H, W]

    return [images_low, images_high]


def non_subsampled_pyramid_recompose(images_low, images_high, g0, g1, level, device="cuda" if torch.cuda.is_available() else "cpu"):
    # 将输入张量移到指定设备
    images_low = torch.tensor(images_low, dtype=torch.float32, device=device) if not isinstance(images_low, torch.Tensor) else images_low.to(device)
    images_high = torch.tensor(images_high, dtype=torch.float32, device=device) if not isinstance(images_high, torch.Tensor) else images_high.to(device)
    g0 = torch.tensor(g0, dtype=torch.float32, device=device) if not isinstance(g0, torch.Tensor) else g0.to(device)
    g1 = torch.tensor(g1, dtype=torch.float32, device=device) if not isinstance(g1, torch.Tensor) else g1.to(device)

    # 验证输入
    if images_low.ndim != 4 or images_high.ndim != 4 or g0.ndim != 2 or g1.ndim != 2:
        raise ValueError("func non_subsampled_pyramid_recompose: images_low, images_high must be 4D [B, C, H, W], g0, g1 must be 2D!")
    if not isinstance(level, int) or level < 1:
        raise ValueError("func non_subsampled_pyramid_recompose: level must be a positive integer!")

    # 获取批次大小和通道数
    B, C, H, W = images_low.shape

    # 递减 level
    level -= 1

    # 分支处理
    if level != 0:
        # 处理 g0
        H_f, W_f = g0.shape
        scale = 2 ** level
        new_g0_shape = (scale * H_f, scale * W_f)
        new_g0 = torch.zeros(new_g0_shape, dtype=torch.float32, device=device)

        # 生成索引（步长为 2^level）
        rows = torch.arange(0, new_g0_shape[0], scale, device=device)
        cols = torch.arange(0, new_g0_shape[1], scale, device=device)
        row_idx, col_idx = torch.meshgrid(rows, cols)
        indices = torch.stack([row_idx.contiguous().view(-1), col_idx.contiguous().view(-1)], dim=1)

        # 向量化填充 g0
        g0_flat = g0.view(-1)
        if indices.shape[0] != g0_flat.shape[0]:
            raise ValueError("func nsp_recompose: g0 size does not match index count!")
        new_g0[indices[:, 0], indices[:, 1]] = g0_flat

        # 裁剪 new_g0
        new_g0 = new_g0[:(H_f - 1) * scale + 1, :(W_f - 1) * scale + 1]

        # 处理 g1
        H_f, W_f = g1.shape
        new_g1_shape = (scale * H_f, scale * W_f)
        new_g1 = torch.zeros(new_g1_shape, dtype=torch.float32, device=device)

        # 重用索引（假设 g0 和 g1 形状相同）
        if g1.shape != g0.shape:
            rows = torch.arange(0, new_g1_shape[0], scale, device=device)
            cols = torch.arange(0, new_g1_shape[1], scale, device=device)
            row_idx, col_idx = torch.meshgrid(rows, cols)
            indices = torch.stack([row_idx.contiguous().view(-1), col_idx.contiguous().view(-1)], dim=1)

        # 向量化填充 g1
        g1_flat = g1.view(-1)
        if indices.shape[0] != g1_flat.shape[0]:
            raise ValueError("func nsp_recompose: g1 size does not match index count!")
        new_g1[indices[:, 0], indices[:, 1]] = g1_flat

        # 裁剪 new_g1
        new_g1 = new_g1[:(H_f - 1) * scale + 1, :(W_f - 1) * scale + 1]
    else:
        new_g0 = g0
        new_g1 = g1

    # 调整滤波器形状以支持多通道卷积
    # 形状 [C, 1, H_f, W_f]：C 个输出通道，每个通道使用相同的单通道滤波器
    new_g0_H, new_g0_W = new_g0.shape
    new_g1_H, new_g1_W = new_g1.shape
    new_g0 = new_g0.unsqueeze(0).unsqueeze(0).expand(C, 1, new_g0_H, new_g0_W)
    new_g1 = new_g1.unsqueeze(0).unsqueeze(0).expand(C, 1, new_g1_H, new_g1_W)

    # 翻转卷积核以匹配 cv2.filter2D
    new_g0 = torch.flip(new_g0, dims=[2, 3])
    new_g1 = torch.flip(new_g1, dims=[2, 3])

    # 计算填充大小
    pad_h = (new_g0.shape[2] - 1) // 2
    pad_w = (new_g0.shape[3] - 1) // 2

    # 应用反射填充
    images_low_padded = F.pad(images_low, (pad_w, pad_w, pad_h, pad_h), mode='reflect')

    # 应用卷积，groups=C 确保每个输入通道独立卷积
    images_low_filtered = F.conv2d(images_low_padded, new_g0, groups=C)  # 形状 [B, C, H, W]

    # 计算填充大小
    pad_h = (new_g1.shape[2] - 1) // 2
    pad_w = (new_g1.shape[3] - 1) // 2

    # 应用反射填充
    images_high_padded = F.pad(images_high, (pad_w, pad_w, pad_h, pad_h), mode='reflect')

    # 应用卷积，groups=C 确保每个输入通道独立卷积
    images_high_filtered = F.conv2d(images_high_padded, new_g1, groups=C)  # 形状 [B, C, H, W]

    # 合并结果
    images_recompose = images_low_filtered + images_high_filtered  # 形状 [B, C, H, W]

    return images_recompose


def non_subsampled_single_filter_bank_decompose(images, f1, f2, mup=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    # 将输入张量移到指定设备
    images = torch.tensor(images, dtype=torch.float32, device=device) if not isinstance(images, torch.Tensor) else images.to(device)
    f1 = torch.tensor(f1, dtype=torch.float32, device=device) if not isinstance(f1, torch.Tensor) else f1.to(device)
    f2 = torch.tensor(f2, dtype=torch.float32, device=device) if not isinstance(f2, torch.Tensor) else f2.to(device)
    mup = torch.tensor(mup, dtype=torch.float32, device=device) if not isinstance(mup, torch.Tensor) and mup is not None else mup

    # 验证输入
    if images.ndim != 4 or f1.ndim != 2 or f2.ndim != 2:
        raise ValueError("func non_subsampled_single_filter_bank_decompose: images must be 4D [B, C, H, W], f1, f2 must be 2D!")
    if mup is not None and not isinstance(mup, torch.Tensor):
        raise ValueError("func non_subsampled_single_filter_bank_decompose: mup must be a tensor or empty list!")
    if mup is not None and mup.shape not in [(1, 1), (2, 2)]:
        raise ValueError("func non_subsampled_single_filter_bank_decompose: mup must be 1x1, 2x2, or empty!")

    # 获取批次大小和通道数
    B, C, H, W = images.shape

    if mup is None or (isinstance(mup, torch.Tensor) and (torch.allclose(mup, torch.tensor(1.0, device=device)) or torch.allclose(mup, torch.eye(2, device=device)))):
        # 直接卷积
        # 调整滤波器形状以支持多通道卷积
        # 形状 [C, 1, H_f, W_f]：C 个输出通道，每个通道使用相同的单通道滤波器
        f1_H, f1_W = f1.shape
        f2_H, f2_W = f2.shape
        f1 = f1.unsqueeze(0).unsqueeze(0).expand(C, 1, f1_H, f1_W)
        f2 = f2.unsqueeze(0).unsqueeze(0).expand(C, 1, f2_H, f2_W)

        # 翻转卷积核以匹配 cv2.filter2D
        f1 = torch.flip(f1, dims=[2, 3])
        f2 = torch.flip(f2, dims=[2, 3])

        # 计算填充大小
        pad_h_f1 = (f1.shape[2] - 1) // 2
        pad_w_f1 = (f1.shape[3] - 1) // 2

        # 应用反射填充
        images_padded_f1 = F.pad(images, (pad_w_f1, pad_w_f1, pad_h_f1, pad_h_f1), mode='reflect')

        # 应用卷积，groups=C 确保每个输入通道独立卷积
        images_low = F.conv2d(images_padded_f1, f1, groups=C)  # 形状 [B, C, H, W]

        # 计算填充大小
        pad_h_f2 = (f2.shape[2] - 1) // 2
        pad_w_f2 = (f2.shape[3] - 1) // 2

        # 应用反射填充
        images_padded_f2 = F.pad(images, (pad_w_f2, pad_w_f2, pad_h_f2, pad_h_f2), mode='reflect')

        # 应用卷积，groups=C 确保每个输入通道独立卷积
        images_high = F.conv2d(images_padded_f2, f2, groups=C)  # 形状 [B, C, H, W]
    else:
        # 使用 zconv_transform（已支持 [B, C, H, W]）
        if mup.shape == (1, 1):
            mup = mup * torch.eye(2, device=device)
        images_low = zconv_transform(images, f1, mup, device=device)  # 形状 [B, C, H, W]
        images_high = zconv_transform(images, f2, mup, device=device)  # 形状 [B, C, H, W]

    return [images_low, images_high]


def non_subsampled_single_filter_bank_recompose(images_low, images_high, f1, f2, mup=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    # 将输入张量移到指定设备
    images_low = torch.tensor(images_low, dtype=torch.float32, device=device) if not isinstance(images_low, torch.Tensor) else images_low.to(device)
    images_high = torch.tensor(images_high, dtype=torch.float32, device=device) if not isinstance(images_high, torch.Tensor) else images_high.to(device)
    f1 = torch.tensor(f1, dtype=torch.float32, device=device) if not isinstance(f1, torch.Tensor) else f1.to(device)
    f2 = torch.tensor(f2, dtype=torch.float32, device=device) if not isinstance(f2, torch.Tensor) else f2.to(device)
    mup = torch.tensor(mup, dtype=torch.float32, device=device) if not isinstance(mup, torch.Tensor) and mup is not None else mup

    # 验证输入
    if images_low.ndim != 4 or images_high.ndim != 4 or f1.ndim != 2 or f2.ndim != 2:
        raise ValueError("func non_subsampled_single_filter_bank_recompose: images_low, images_high must be 4D [B, C, H, W], f1, f2 must be 2D!")
    if mup is not None and not isinstance(mup, torch.Tensor):
        raise ValueError("func non_subsampled_single_filter_bank_recompose: mup must be a tensor or empty list!")
    if mup is not None and mup.shape != (2, 2):
        raise ValueError("func non_subsampled_single_filter_bank_recompose: mup must be 2x2 or empty!")

    # 获取批次大小和通道数
    B, C, H, W = images_low.shape

    if mup is None:
        # 直接卷积
        # 调整滤波器形状以支持多通道卷积
        # 形状 [C, 1, H_f, W_f]：C 个输出通道，每个通道使用相同的单通道滤波器
        f1_H, f1_W = f1.shape
        f2_H, f2_W = f2.shape
        f1 = f1.unsqueeze(0).unsqueeze(0).expand(C, 1, f1_H, f1_W)
        f2 = f2.unsqueeze(0).unsqueeze(0).expand(C, 1, f2_H, f2_W)

        # 翻转卷积核以匹配 cv2.filter2D
        f1 = torch.flip(f1, dims=[2, 3])
        f2 = torch.flip(f2, dims=[2, 3])

        # 计算填充大小
        pad_h_f1 = (f1.shape[2] - 1) // 2
        pad_w_f1 = (f1.shape[3] - 1) // 2

        # 应用反射填充
        images_low_padded_f1 = F.pad(images_low, (pad_w_f1, pad_w_f1, pad_h_f1, pad_h_f1), mode='reflect')

        # 应用卷积，groups=C 确保每个输入通道独立卷积
        y1 = F.conv2d(images_low_padded_f1, f1, groups=C)  # 形状 [B, C, H, W]

        # 计算填充大小
        pad_h_f2 = (f2.shape[2] - 1) // 2
        pad_w_f2 = (f2.shape[3] - 1) // 2

        # 应用反射填充
        images_high_padded_f2 = F.pad(images_high, (pad_w_f2, pad_w_f2, pad_h_f2, pad_h_f2), mode='reflect')

        # 应用卷积，groups=C 确保每个输入通道独立卷积
        y2 = F.conv2d(images_high_padded_f2, f2, groups=C)  # 形状 [B, C, H, W]
    else:
        # 使用 zconv_transform（已支持 [B, C, H, W]）
        y1 = zconv_transform(images_low, f1, mup, device=device)  # 形状 [B, C, H, W]
        y2 = zconv_transform(images_high, f2, mup, device=device)  # 形状 [B, C, H, W]

    # 合并结果
    images_out = y1 + y2  # 形状 [B, C, H, W]

    return images_out


def non_subsampled_directional_filter_bank_decompose(images, dfilter, num_levels, device="cuda" if torch.cuda.is_available() else "cpu"):
    # 将输入张量移到指定设备
    images = torch.tensor(images, dtype=torch.float32, device=device) if not isinstance(images, torch.Tensor) else images.to(device)

    # 提取滤波器
    k1, k2, f1, f2 = dfilter

    # 定义变换矩阵 q1
    q1 = torch.tensor([[1, -1], [1, 1]], dtype=torch.float32, device=device)

    # 初始化输出
    y = [[], [], [], []]

    # 分支处理
    if num_levels == 1:
        # 第一层分解，输出 [B, C, H, W]
        y[0], y[1] = non_subsampled_single_filter_bank_decompose(images, k1, k2, None, device=device)
    else:
        # 第一层分解
        x1, x2 = non_subsampled_single_filter_bank_decompose(images, k1, k2, None, device=device)

        # 第二层分解
        y[0], y[1] = non_subsampled_single_filter_bank_decompose(x1, k1, k2, q1, device=device)
        y[2], y[3] = non_subsampled_single_filter_bank_decompose(x2, k1, k2, q1, device=device)

        # 后续层分解
        for l in range(3, num_levels + 1):
            y_old = y
            y = [[] for _ in range(2 ** l)]

            for k in range(2 ** (l - 2)):
                slk = 2 * (k // 2) - 2 ** (l - 3) + 1
                mkl = 2 * torch.matmul(
                    torch.tensor([[2 ** (l - 3), 0], [0, 1]], dtype=torch.float32, device=device),
                    torch.tensor([[1, 0], [-slk, 1]], dtype=torch.float32, device=device)
                )
                i = k % 2
                y[2 * k], y[2 * k + 1] = non_subsampled_single_filter_bank_decompose(y_old[k], f1[i], f2[i], mkl, device=device)

            for k in range(2 ** (l - 2), 2 ** (l - 1)):
                slk = 2 * ((k - 2 ** (l - 2) - 1) // 2) - 2 ** (l - 3) + 1
                mkl = 2 * torch.matmul(
                    torch.tensor([[1, 0], [0, 2 ** (l - 3)]], dtype=torch.float32, device=device),
                    torch.tensor([[1, -slk], [0, 1]], dtype=torch.float32, device=device)
                )
                i = (k % 2) + 2
                y[2 * k], y[2 * k + 1] = non_subsampled_single_filter_bank_decompose(y_old[k], f1[i], f2[i], mkl, device=device)

    return y


def non_subsampled_directional_filter_bank_recompose(subbands, dfilter, device="cuda" if torch.cuda.is_available() else "cpu"):
    # 将输入子带张量移到指定设备
    subbands = [
        torch.tensor(s, dtype=torch.float32, device=device) if not isinstance(s, torch.Tensor) and s != [] else s.to(device) if isinstance(s, torch.Tensor) else s
        for s in subbands
    ]

    # 将滤波器移到指定设备
    dfilter = [
        f if isinstance(f, list) else torch.tensor(f, dtype=torch.float32, device=device) if not isinstance(f, torch.Tensor) else f.to(
            device)
        for f in dfilter
    ]
    for i, f in enumerate(dfilter):
        if isinstance(f, list):
            dfilter[i] = [
                torch.tensor(fi, dtype=torch.float32, device=device) if not isinstance(fi, torch.Tensor) else fi.to(device) for fi in f]

    # 计算非空子带数量
    len_x = sum(1 for s in subbands if s != [])
    if len_x == 0 or (len_x & (len_x - 1)) != 0:  # 验证 len_x 是 2 的幂
        raise ValueError("func non_subsampled_directional_filter_bank_recompose: number of non-empty subbands must be a power of 2!")

    # 计算层数
    num_levels = int(math.log2(len_x))

    # 提取滤波器
    k1, k2, f1, f2 = dfilter

    # 定义变换矩阵 q1
    q1 = torch.tensor([[1, -1], [1, 1]], dtype=torch.float32, device=device)

    # 分支处理
    if num_levels == 1:
        # 第一层重构，输出 [B, C, H, W]
        images_out = non_subsampled_single_filter_bank_recompose(subbands[0], subbands[1], k1, k2, None, device=device)
    else:
        x = subbands.copy()  # 复制以修改
        for l in range(num_levels, 2, -1):
            for k in range(2 ** (l - 2)):
                slk = 2 * (k // 2) - 2 ** (l - 3) + 1
                mkl = 2 * torch.matmul(
                    torch.tensor([[2 ** (l - 3), 0], [0, 1]], dtype=torch.float32, device=device),
                    torch.tensor([[1, 0], [-slk, 1]], dtype=torch.float32, device=device)
                )
                i = k % 2
                x[k] = non_subsampled_single_filter_bank_recompose(x[2 * k], x[2 * k + 1], f1[i], f2[i], mkl, device=device)

            for k in range(2 ** (l - 2), 2 ** (l - 1)):
                slk = 2 * ((k - 2 ** (l - 2) - 1) // 2) - 2 ** (l - 3) + 1
                mkl = 2 * torch.matmul(
                    torch.tensor([[1, 0], [0, 2 ** (l - 3)]], dtype=torch.float32, device=device),
                    torch.tensor([[1, -slk], [0, 1]], dtype=torch.float32, device=device)
                )
                i = (k % 2) + 2
                x[k] = non_subsampled_single_filter_bank_recompose(x[2 * k], x[2 * k + 1], f1[i], f2[i], mkl, device=device)

        x[0] = non_subsampled_single_filter_bank_recompose(x[0], x[1], k1, k2, q1, device=device)
        x[1] = non_subsampled_single_filter_bank_recompose(x[2], x[3], k1, k2, q1, device=device)
        images_out = non_subsampled_single_filter_bank_recompose(x[0], x[1], k1, k2, None, device=device)

    return images_out


def non_subsampled_contourlet_decompose(images, levels, pfilter_name, dfilter_name, device="cuda" if torch.cuda.is_available() else "cpu"):
    # 将输入张量移到指定设备
    images = torch.tensor(images, dtype=torch.float32, device=device) if not isinstance(images, torch.Tensor) else images.to(device)
    # 获取金字塔滤波器
    ph0, ph1, _, _ = atrous_filters(pfilter_name, device=device)

    # filtersd = np.zeros((4))
    filtersd = [[], [], [], []]

    # 初始化方向滤波器
    direction_filters = [[], [], [], []]
    dh1, dh2 = dfilters(dfilter_name, 'd', device=device)
    dh1 = dh1 / torch.sqrt(torch.tensor(2.0, device=device))
    dh2 = dh2 / torch.sqrt(torch.tensor(2.0, device=device))

    direction_filters[0] = modulate2(dh1, 'c', None, device=device)
    direction_filters[1] = modulate2(dh2, 'c', None, device=device)
    direction_filters[2], direction_filters[3] = pfilters(dh1, dh2, device=device)

    # 初始化输出
    num_levels = len(levels)
    n_index = num_levels + 1
    contourlet_coeffs = [[] for _ in range(n_index)]
    pyramid_coeffs = [[] for _ in range(num_levels)]

    # 分解过程
    current_images = images
    for i in range(num_levels):
        images_low, images_high = non_subsampled_pyramid_decompose(current_images, ph0, ph1, i, device=device)

        if levels[n_index - 2] > 0:
            images_high_dir = non_subsampled_directional_filter_bank_decompose(images_high, direction_filters, levels[n_index - 2], device=device)
            contourlet_coeffs[n_index - 1] = images_high_dir
        else:
            contourlet_coeffs[n_index - 1] = images_high

        n_index -= 1
        current_images = images_low
        pyramid_coeffs[i] = current_images

    contourlet_coeffs[0] = current_images

    return pyramid_coeffs, contourlet_coeffs


def non_subsampled_contourlet_recompose(contourlet_coeffs, levels, pfilter_name, dfilter_name, device="cuda" if torch.cuda.is_available() else "cpu"):
    # 获取金字塔滤波器
    _, _, pg0, pg1 = atrous_filters(pfilter_name, device=device)

    # 初始化方向滤波器
    direction_filters = [[], [], [], []]
    dg1, dg2 = dfilters(dfilter_name, 'r', device=device)
    dg1 = dg1 / torch.sqrt(torch.tensor(2.0, device=device))
    dg2 = dg2 / torch.sqrt(torch.tensor(2.0, device=device))

    direction_filters[0] = modulate2(dg1, 'c', None, device=device)
    direction_filters[1] = modulate2(dg2, 'c', None, device=device)
    direction_filters[2], direction_filters[3] = pfilters(dg1, dg2, device=device)

    # 重构过程
    num_levels = len(levels)
    images_low = contourlet_coeffs[0]

    for i in range(num_levels):
        if isinstance(contourlet_coeffs[i + 1], list) and len(contourlet_coeffs[i + 1]) > 1:
            images_high = non_subsampled_directional_filter_bank_recompose(
                contourlet_coeffs[i + 1], direction_filters, device=device
            )
        else:
            images_high = contourlet_coeffs[i + 1]

        images_low = non_subsampled_pyramid_recompose(
            images_low, images_high, pg0, pg1, num_levels - i, device=device
        )

    images_out = images_low
    return images_out

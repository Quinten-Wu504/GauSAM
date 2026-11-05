import torch
import math
import random
import albumentations as A


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers."""
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs,indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def make_odd(num):
    num = math.ceil(num)
    if num % 2 == 0:
        num += 1
    return num

def apply_transform(image, mask):
    strategy = [(1, 2), (0, 3), (0, 2), (1, 1)]  # 各种组合的数量配置
    level = 5  # 变换强度级别

    # 定义变换列表
    transform = A.Compose([
        A.ColorJitter(brightness=0.04 * level, contrast=0, saturation=0, hue=0, p=0.2 * level),
        A.ColorJitter(brightness=0, contrast=0.04 * level, saturation=0, hue=0, p=0.2 * level),
        A.Posterize(num_bits=math.floor(8 - 0.8 * level), p=0.2 * level),
        A.Sharpen(alpha=(0.04 * level, 0.1 * level), lightness=(1, 1), p=0.2 * level),
        A.GaussianBlur(blur_limit=(3, make_odd(3 + 0.8 * level)), p=0.2 * level),
        A.GaussNoise(
            std_range=(0.02 * level, 0.1 * level),
            mean_range=(0, 0),
            per_channel=True,
            noise_scale_factor=1,
            p=0.2 * level
        ),
        A.Rotate(limit=4 * level, interpolation=1, border_mode=0, rotate_method='largest_box', crop_border=False, p=0.2 * level),
        A.HorizontalFlip(p=0.2 * level),
        A.VerticalFlip(p=0.2 * level),
        A.Affine(scale=(1 - 0.04 * level, 1 + 0.04 * level), interpolation=1, mask_interpolation=0, p=0.2 * level),
        A.Affine(shear={'x': (0, 2 * level), 'y': (0, 0)}, interpolation=1, mask_interpolation=0, p=0.2 * level),  # x 方向剪切
        A.Affine(shear={'x': (0, 0), 'y': (0, 2 * level)}, interpolation=1, mask_interpolation=0, p=0.2 * level),  # y 方向剪切
        A.Affine(translate_percent={'x': (0, 0.02 * level), 'y': (0, 0)}, interpolation=1, mask_interpolation=0, p=0.2 * level),  # x 方向平移
        A.Affine(translate_percent={'x': (0, 0), 'y': (0, 0.02 * level)}, interpolation=1, mask_interpolation=0, p=0.2 * level),  # y 方向平移
    ])

    # 随机选择一个策略
    employ = random.choice(strategy)

    # 选择对应数量的变换
    level_transforms = random.sample(transform.transforms[:6], employ[0])
    shape_transforms = random.sample(transform.transforms[6:], employ[1])
    img_transform = A.Compose(level_transforms + shape_transforms, seed=random.randrange(2**32))
    random.shuffle(img_transform.transforms)
    transformed = img_transform(image=image, mask=mask)

    return transformed['image'], transformed['mask']

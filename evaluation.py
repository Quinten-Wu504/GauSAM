"""Per-image foreground Dice and symmetric maximum HD in output pixels.
One-sided empty masks score (0, inf); both-empty masks score (1, 0).
"""
import os
import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from monai.metrics import compute_hausdorff_distance


class DistributedEvalSampler(Sampler):
    """Shard evaluation without padding/duplicating samples."""
    def __init__(self, dataset, num_replicas=None, rank=None):
        self.dataset = dataset
        self.num_replicas = dist.get_world_size() if num_replicas is None else num_replicas
        self.rank = dist.get_rank() if rank is None else rank

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        return len(range(self.rank, len(self.dataset), self.num_replicas))


def compute_dice_and_hd(pred_masks, gt, device=None):
    if pred_masks.shape != gt.shape or pred_masks.ndim != 4 or pred_masks.shape[1] != 1:
        raise ValueError("Metrics require matching [B,1,H,W] foreground masks")
    scores = []
    for pred, truth in zip(pred_masks.bool(), gt.bool()):
        ps, ts = pred.sum(), truth.sum()
        if ps == 0 and ts == 0:
            scores.append((pred_masks.new_tensor(1.), pred_masks.new_tensor(0.)))
        else:
            dice = 2.0 * (pred & truth).sum() / (ps + ts)
            hd = (pred_masks.new_tensor(float('inf')) if ps == 0 or ts == 0 else
                  compute_hausdorff_distance(pred[None].float(), truth[None].float(),
                                            include_background=True, directed=False).squeeze())
            scores.append((dice, hd))
    return tuple(torch.stack([s[i] for s in scores]).mean() for i in range(2))


def discretize(logits, gt):
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("Expected foreground logits [B,1,H,W]")
    truth = gt.unsqueeze(1) if gt.ndim == 3 else gt
    return (logits.sigmoid() > 0.5).float(), (truth > 0).float()


@torch.no_grad()
def evaluate(model, val_loader, sam_trans, device, args):
    model.eval()
    distributed = dist.is_available() and dist.is_initialized()
    # Uneven eval shards must not invoke DDP's per-forward collectives.
    net = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    if distributed:
        for buffer in net.buffers():
            dist.broadcast(buffer, src=0)
    sums = torch.zeros(3, device=device, dtype=torch.float64)
    for image, boxes, gt, _, img_ids, gt_ids in val_loader:
        box = sam_trans.apply_boxes(boxes.numpy(), tuple(gt.shape[-2:]))
        box = torch.as_tensor(box, dtype=torch.float32, device=device)
        if box.ndim == 2:
            box = box[:, None, :]
        image, gt = image.to(device), gt.to(device)
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        logits = net(image, box, gt.shape[-2:])
        pred, truth = discretize(logits, gt)
        for i in range(image.shape[0]):
            dice, hd = compute_dice_and_hd(pred[i:i+1], truth[i:i+1])
            sums[0] += dice
            sums[1] += hd
            sums[2] += 1
            if getattr(args, 'save_pic', False):
                out_dir = os.path.join(args.save_dir, 'predictions')
                os.makedirs(out_dir, exist_ok=True)
                mask = pred[i, 0] * 255
                path = os.path.join(out_dir, os.path.splitext(os.path.basename(img_ids[i]))[0] + '.png')
                if not cv2.imwrite(path, mask.cpu().numpy().astype(np.uint8)):
                    raise OSError(f"Could not save {path}")
    if distributed:
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    if sums[2] == 0:
        raise ValueError("Evaluation dataset is empty")
    return (sums[0] / sums[2]).item(), (sums[1] / sums[2]).item()

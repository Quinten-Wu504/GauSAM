"""Train and evaluate GauSAM. Run `python main.py --help` for the CLI."""
import argparse
import json
import logging
import math
import os
from pathlib import Path
import random
import sys


ARCHITECTURE = (
    "model_type", "image_size", "n_class", "n_feat", "hidden_dim",
    "hidden_list", "lora_rank", "no_upsampling", "decoder_mode",
)
RESUME_FIXED = ARCHITECTURE + (
    "dataset", "label_size", "epochs", "lr", "min_lr", "lora_lr_ratio",
    "weight_decay", "batch_size", "num_workers", "query_chunk_size", "seed", "deterministic", "grad_clip",
    "eval_freq", "patience", "min_delta",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", help="JSON configuration; explicit CLI options take precedence")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--test_only", action="store_true", help="Alias for --mode eval")
    parser.add_argument("--dataset", choices=["polyp", "cvc"], default="polyp")
    parser.add_argument("--data_path", "--dataset_path", required=False, help="Existing prepared dataset directory; omit to use the automatic default dataset")
    parser.add_argument("--split", choices=["val", "test"], default="test", help="Evaluation split")
    parser.add_argument("--work_dir", default="outputs")
    parser.add_argument("--task_name", default="gausam", help="Run name under work_dir")
    parser.add_argument("--resume", help="Full GauSAM checkpoint: resume training or evaluate")
    parser.add_argument("--cache_dir", default="data/.cache", help="Reusable download cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size per process")
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1000, help="Total epochs, including resumed epochs")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Final base LR; LoRA retains its LR ratio")
    parser.add_argument("--lora_lr_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=0.0, help="Gradient norm limit; 0 disables")
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--save_freq", type=int, default=100, help="Periodic checkpoint interval")
    parser.add_argument("--patience", type=int, default=20, help="Validation events without improvement; 0 disables")
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2041)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model_type", choices=["vit_b", "vit_l", "vit_h"], default="vit_b")
    parser.add_argument("--image_size", type=int, default=None, help="SAM input size; defaults to 384")
    parser.add_argument("--label_size", "--output_size", type=int, default=None, help="Output/mask size; defaults to input size")
    parser.add_argument("--n_class", type=int, default=500, help="Gaussian template count")
    parser.add_argument("--n_feat", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--hidden_list", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--no_upsampling", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--decoder_mode", choices=["field", "frequency"], default="frequency")
    parser.add_argument("--query_chunk_size", type=int, default=256, help="Gaussian queries per chunk")
    parser.add_argument("--save_pic", action="store_true", help="Save label PNGs during evaluation")
    parser.add_argument("--print_config", action="store_true", help="Print CLI/config-file settings and exit without importing PyTorch")
    argv = sys.argv[1:] if argv is None else argv
    preliminary, _ = parser.parse_known_args(argv)
    config = {}
    if preliminary.config:
        with open(preliminary.config, encoding="utf-8") as stream:
            config = json.load(stream)
        known = {action.dest for action in parser._actions}
        if not isinstance(config, dict) or set(config) - known:
            parser.error("Configuration must be a JSON object containing recognized argument names")
        parser.set_defaults(**config)
    args = parser.parse_args(argv)
    explicit = set(config)
    for action in parser._actions:
        if any(token.split("=", 1)[0] in action.option_strings for token in argv):
            explicit.add(action.dest)
    if args.test_only:
        args.mode = "eval"
    if not args.task_name or Path(args.task_name).name != args.task_name or args.task_name in (".", ".."):
        parser.error("task_name must be a single directory name")
    return args, explicit


def finalize_config(args):
    if args.dataset not in ("polyp", "cvc") or args.mode not in ("train", "eval"):
        raise ValueError("Invalid dataset or mode")
    args.image_size = 384 if args.image_size is None else args.image_size
    args.label_size = args.image_size if args.label_size is None else args.label_size
    positive = ("image_size", "label_size", "n_class", "n_feat", "hidden_dim",
                "lora_rank", "query_chunk_size", "batch_size", "eval_batch_size", "epochs", "eval_freq", "save_freq")
    if any(not isinstance(getattr(args, key), int) or getattr(args, key) <= 0 for key in positive):
        raise ValueError("Sizes, counts and intervals must be positive integers")
    if args.image_size % 16 or args.image_size < 64:
        raise ValueError("image_size must be at least 64 and divisible by SAM's patch size (16)")
    if not isinstance(args.hidden_list, list) or not args.hidden_list or any(not isinstance(x, int) or x <= 0 for x in args.hidden_list):
        raise ValueError("hidden_list must contain positive integers")
    if args.num_workers < 0 or args.patience < 0 or args.grad_clip < 0 or args.min_delta < 0:
        raise ValueError("Workers, patience, clipping and min_delta must be nonnegative")
    if not 0 < args.min_lr <= args.lr or args.lora_lr_ratio <= 0 or args.weight_decay < 0:
        raise ValueError("Require 0 < min_lr <= lr, lora_lr_ratio > 0 and weight_decay >= 0")
    if args.mode == "train" and args.label_size != args.image_size:
        raise ValueError("Training augmentations require label_size=image_size; use --mode eval for cross-resolution queries")
    if args.model_type not in ("vit_b", "vit_l", "vit_h") or args.decoder_mode not in ("field", "frequency"):
        raise ValueError("Invalid model_type or decoder_mode")
    for key in ("deterministic", "no_upsampling", "save_pic", "test_only", "print_config"):
        if not isinstance(getattr(args, key), bool):
            raise ValueError(f"{key} must be a JSON boolean")
    if not all(math.isfinite(getattr(args, key)) for key in ("lr", "min_lr", "lora_lr_ratio", "weight_decay", "grad_clip", "min_delta")):
        raise ValueError("Optimization parameters must be finite")
    if args.dataset == "cvc" and args.mode != "eval":
        raise ValueError("CVC is an evaluation dataset; train on Sessile-Kvasir")
    if args.dataset == "cvc" and args.split != "test":
        raise ValueError("CVC evaluation uses --split test")
    if not isinstance(args.seed, int) or not 0 <= args.seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")


def restore_config(args, explicit, checkpoint):
    if checkpoint.get("implementation_version") != 2 or "config" not in checkpoint:
        raise ValueError("Expected a corrected GauSAM implementation_version=2 checkpoint; legacy weights require retraining")
    saved = dict(checkpoint["config"])
    if isinstance(saved.get("hidden_list"), str):
        saved["hidden_list"] = json.loads(saved["hidden_list"])
    if args.mode == "train" and checkpoint.get("runner_version") != 1:
        raise ValueError("This checkpoint supports evaluation only; full resume needs a main.py checkpoint with optimizer and scheduler state")
    protected = RESUME_FIXED if args.mode == "train" else ARCHITECTURE
    for key in protected:
        if key in saved:
            if key in explicit and getattr(args, key) != saved[key]:
                raise ValueError(f"{key} conflicts with checkpoint ({getattr(args, key)!r} vs {saved[key]!r}); omit this option")
            setattr(args, key, saved[key])
    if "dataset" not in explicit:
        args.dataset = saved.get("dataset", "polyp")


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    import numpy as np
    import torch
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    import cv2
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)


def build_model(args, device, pretrained=True):
    from model.segment_anything import sam_model_registry
    from model.sam_encoder import SamEncoder
    from model.ada_gaussian import AdaGsSplatting
    from model.GauSAM import GauSAM
    sam = sam_model_registry[args.model_type](image_size=args.image_size,
                                            checkpoint=args.checkpoint if pretrained else None)
    encoder = SamEncoder(sam, r=args.lora_rank, n_class=args.n_class,
                         n_feat=args.n_feat, no_upsampling=args.no_upsampling)
    gaussian = AdaGsSplatting(kernel_size=3, encoder_dim=args.n_feat,
                             hidden_dim=args.hidden_dim, num_points=args.n_class,
                             decoder_mode=args.decoder_mode, query_chunk_size=args.query_chunk_size)
    return GauSAM(encoder, gaussian, hidden_list=args.hidden_list).to(device)


def build_dataset(args, split):
    size, label = (args.image_size,) * 2, (args.label_size,) * 2
    if split in ("train", "val"):
        from dataset.my_dataset import MyDataset
        dataset = MyDataset(args.data_path, train=split == "train", resize_size=size,
                            label_resize_size=label)
        pairs = zip(dataset.img_files, dataset.gt_files)
    else:
        from dataset.test_dataset import TestDataset
        test_root = Path(args.data_path) / "test"
        if args.dataset == "cvc" and not test_root.is_dir():
            test_root = Path(args.data_path)
        dataset = TestDataset(test_root, resize_size=size,
                              label_resize_size=label)
        pairs = zip(dataset.img_files, dataset.gt_files)
    if not len(dataset):
        raise ValueError(f"Empty {split} dataset at {args.data_path}")
    for image, mask in pairs:
        if not Path(image).is_file() or not Path(mask).is_file():
            raise FileNotFoundError(f"Missing image/mask pair: {image}, {mask}")
    return dataset


def json_write(path, content):
    # Infinity denotes an empty-mask failure; use an explicit string in valid JSON.
    def clean(value):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        return value
    Path(path).write_text(json.dumps(clean(content), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(args, explicit):
    # Must be configured before CUDA context creation, including checkpoint loading.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from evaluation import DistributedEvalSampler, evaluate
    from model.segment_anything.utils.transforms import ResizeLongestSide

    state = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if state is not None:
        restore_config(args, explicit, state)
    finalize_config(args)
    if args.mode == "eval" and state is None:
        raise ValueError("Evaluation requires --resume pointing to a trained GauSAM checkpoint; SAM weights alone cannot evaluate GauSAM")
    from prepare import prepare_inputs
    prepare_inputs(args)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    device = torch.device(args.device)
    if world > 1:
        if device.type != "cuda":
            raise ValueError("Distributed execution requires CUDA and torchrun")
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
        dist.init_process_group("nccl")
    if state is not None and args.mode == "train" and state["world_size"] != world:
        raise ValueError("Resume with the original world size to preserve batch/sampler semantics")
    output = Path(args.work_dir) / args.task_name
    # Rank zero checks before any rank writes, avoiding a directory-creation race.
    occupied = rank == 0 and args.mode == "train" and state is None and output.exists() and any(output.iterdir())
    occupied = torch.tensor(int(occupied), device=device)
    if world > 1:
        dist.broadcast(occupied, src=0)
    if occupied.item():
        raise FileExistsError(f"Run directory is not empty: {output}; use a new task_name or --resume")
    output.mkdir(parents=True, exist_ok=True)
    args.save_dir = str(output)
    logger = logging.getLogger("gausam")
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    logger.propagate = False
    logger.addHandler(logging.StreamHandler())
    if rank == 0:
        handler = logging.FileHandler(output / f"{args.mode}.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
        json_write(output / f"{args.mode}_config.json", vars(args))
        from prepare import dataset_fingerprint
        splits = ("train", "val") if args.mode == "train" else (args.split,)
        import hashlib
        source_hash = hashlib.sha256()
        for path in sorted(Path(__file__).parent.rglob("*.py")):
            if not any(part.startswith(".") for part in path.relative_to(Path(__file__).parent).parts):
                source_hash.update(path.relative_to(Path(__file__).parent).as_posix().encode())
                source_hash.update(path.read_bytes())
        json_write(output / f"{args.mode}_provenance.json", {
            "data_sha256": dataset_fingerprint(args.data_path, args.dataset, splits),
            "splits": list(splits), "source_sha256": source_hash.hexdigest(),
        })
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = args.deterministic
    torch.use_deterministic_algorithms(args.deterministic, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import cv2
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    if rank == 0:
        import platform
        from importlib.metadata import version
        json_write(output / "environment.json", {
            "python": platform.python_version(), "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
            "world_size": world, "deterministic": args.deterministic,
            "packages": {name: version(name) for name in ("numpy", "albumentations", "monai", "torchvision")},
        })
    logger.info("Seed: %d | deterministic: %s | batch size: %d | workers: %d",
                args.seed, args.deterministic, args.batch_size, args.num_workers)
    model = build_model(args, device, pretrained=state is None)
    if state is not None:
        model.load_state_dict(state["model_state_dict"], strict=True)
    if world > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    transform = ResizeLongestSide(args.image_size)
    logger.info("Parameters: %s total, %s trainable", sum(p.numel() for p in model.parameters()),
                sum(p.numel() for p in model.parameters() if p.requires_grad))
    generator = torch.Generator().manual_seed(args.seed)
    eval_generator = torch.Generator().manual_seed(args.seed)

    def loader(split, training=False):
        dataset = build_dataset(args, split)
        sampler = (DistributedSampler(dataset, shuffle=True, seed=args.seed) if training
                   else DistributedEvalSampler(dataset)) if world > 1 else None
        logger.info("%s: %d images", split, len(dataset))
        return DataLoader(dataset, batch_size=args.batch_size if training else args.eval_batch_size,
                          shuffle=training and sampler is None, sampler=sampler,
                          num_workers=args.num_workers, pin_memory=device.type == "cuda",
                          worker_init_fn=seed_worker, generator=generator if training else eval_generator, drop_last=False)

    if args.mode == "eval":
        scores = evaluate(model, loader(args.split), transform, device, args)
        if rank == 0:
            json_write(output / "metrics.json", {"dice": scores[0], "hd": scores[1],
                       "split": args.split, "checkpoint": args.resume, "label_size": args.label_size,
                       "hd_unit": "resized-image pixels", "prompt": "ground-truth bounding box"})
            logger.info("Dice: %.6f | HD: %.6f", *scores)
        return

    import monai
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm
    train_loader, val_loader = loader("train", True), loader("val")
    groups = [[p for name, p in model.named_parameters() if p.requires_grad and ("lora" in name) == adapted]
              for adapted in (True, False)]
    optimizer = torch.optim.AdamW([{"params": groups[0], "lr": args.lr * args.lora_lr_ratio},
                                  {"params": groups[1], "lr": args.lr}], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0,
                                                  end_factor=args.min_lr / args.lr, total_iters=args.epochs)
    criterion = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True,
                                      lambda_dice=0.8, lambda_ce=0.2)
    start, best, best_loss, best_hd, stale, early_best = 0, -math.inf, math.inf, math.inf, 0, -math.inf
    if state is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start = state["epoch"] + 1
        best, best_loss, best_hd = state["best_dice"], state["best_loss"], state["best_hd"]
        stale, early_best = state["stale_validations"], state["early_best"]
    del state
    if start >= args.epochs or (args.patience and stale >= args.patience):
        logger.info("Checkpoint already reached the epoch budget or early-stopping condition")
        return
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index],
                                                          find_unused_parameters=True)
    net = model.module if world > 1 else model
    writer = SummaryWriter(str(output / "tensorboard")) if rank == 0 else None

    def save(epoch, name):
        if rank != 0:
            return
        payload = {"implementation_version": 2, "runner_version": 1, "config": vars(args),
                   "model_state_dict": net.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                   "scheduler_state_dict": scheduler.state_dict(), "epoch": epoch, "world_size": world,
                   "best_dice": best, "best_loss": best_loss, "best_hd": best_hd,
                   "stale_validations": stale, "early_best": early_best}
        destination = output / f"checkpoint_{name}.pth"
        temporary = destination.with_suffix(".pth.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, destination)

    try:
        for epoch in range(start, args.epochs):
            # Epoch-specific seeds make epoch-boundary resume independent of earlier RNG consumption.
            epoch_seed = args.seed + epoch * world + rank
            seed_everything(epoch_seed)
            generator.manual_seed(epoch_seed)
            eval_generator.manual_seed(epoch_seed)
            if world > 1:
                train_loader.sampler.set_epoch(epoch)
            model.train()
            totals = torch.zeros(2, dtype=torch.float64, device=device)
            used_lr = optimizer.param_groups[-1]["lr"]
            for image, boxes, gt, _, _, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", disable=rank != 0):
                box = torch.as_tensor(transform.apply_boxes(boxes.numpy(), tuple(gt.shape[-2:])),
                                      dtype=torch.float32, device=device)[:, None, :]
                image, gt = image.to(device), gt.to(device)
                if image.shape[1] == 1:
                    image = image.repeat(1, 3, 1, 1)
                optimizer.zero_grad(set_to_none=True)
                logits = model(image, box, gt.shape[-2:])
                target = gt.float()
                loss = criterion(logits, target)
                finite = torch.isfinite(loss).to(torch.int32)
                if world > 1:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise FloatingPointError("Non-finite loss; checkpoint_latest contains the last completed epoch")
                loss.backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
                optimizer.step()
                totals[0] += loss.detach() * image.shape[0]
                totals[1] += image.shape[0]
            if world > 1:
                dist.all_reduce(totals)
            train_loss = (totals[0] / totals[1]).item()
            scheduler.step()
            improved_loss = train_loss < best_loss
            best_loss = min(best_loss, train_loss)
            row = {"epoch": epoch + 1, "train_loss": train_loss, "lr": used_lr}
            improved_dice = improved_hd = False
            if (epoch + 1) % args.eval_freq == 0 or epoch + 1 == args.epochs:
                # Validation never writes prediction PNGs into the training run.
                eval_args = argparse.Namespace(**vars(args))
                eval_args.save_pic = False
                dice, hd = evaluate(model, val_loader, transform, device, eval_args)
                row.update(dice=dice, hd=hd)
                improved_dice, improved_hd = dice > best, hd < best_hd
                best, best_hd = max(best, dice), min(best_hd, hd)
                if dice > early_best + args.min_delta:
                    early_best, stale = dice, 0
                else:
                    stale += 1
            if rank == 0:
                logger.info("Epoch %d: %s", epoch + 1, row)
                # One file per epoch is avoided: JSON Lines can be read with pandas.read_json(lines=True).
                serializable = {k: str(v) if isinstance(v, float) and not math.isfinite(v) else v for k, v in row.items()}
                with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(serializable, allow_nan=False) + "\n")
                for key, value in row.items():
                    if key != "epoch":
                        writer.add_scalar(key, value, epoch + 1)
            save(epoch, "latest")
            for improved, name in ((improved_loss, "best_loss"), (improved_dice, "best_dice"), (improved_hd, "best_hd")):
                if improved:
                    save(epoch, name)
            if (epoch + 1) % args.save_freq == 0:
                save(epoch, f"epoch_{epoch + 1:04d}")
            if args.patience and stale >= args.patience:
                logger.info("Early stopping after %d validation events without improvement", stale)
                break
        logger.info("Training finished. Best validation Dice: %.6f", best)
    finally:
        if writer is not None:
            writer.close()


def main(argv=None):
    args, explicit = parse_args(argv)
    if args.print_config:
        finalize_config(args)
        print(json.dumps(vars(args), indent=2))
        return
    # Python reads its hash seed at interpreter startup, before CUDA is initialized.
    if argv is None and os.environ.get("PYTHONHASHSEED") != str(args.seed):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = str(args.seed)
        os.execve(sys.executable, [sys.executable, *sys.argv], environment)
    try:
        run(args, explicit)
    finally:
        # Also clean up a process group when a run raises an exception.
        if "torch.distributed" in sys.modules:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    main()

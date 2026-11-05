"""Download and prepare runtime inputs before model construction or DDP startup."""
import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import shutil
import ssl
import stat
import tempfile
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import zipfile

from filelock import FileLock

LOG = logging.getLogger("gausam.prepare")
SAM_FILES = {"vit_b": "sam_vit_b_01ec64.pth", "vit_l": "sam_vit_l_0b3195.pth", "vit_h": "sam_vit_h_4b8939.pth"}
# Public source archives. A mirror must preserve these exact ZIP bytes.
DATASETS = {
    "polyp": {"directory": "sessile-Kvasir", "archive": "sessile-Kvasir.zip",
              "url": "https://drive.google.com/file/d/1uQ6fTLht97EDudSxcVljpQwixUjJDQqP/view",
              "sha256": "b0c00c8808d0a42ebf244301b5098c82b68236e1cb9bb0dcbdfeef025f0f21d0",
              "inventory": "de54bc84a1aa3d255dc5175ac10c6282b99dca399cd8d1667cff6b45be7436b6"},
    "cvc": {"directory": "CVC", "archive": "CVC.zip",
            "url": "https://drive.google.com/file/d/1aoFKT6yn4ZhGMkGPLYSVtwk3cJnXuCum/view",
            "sha256": "13a40852b4f6c244e00aad96be8917fa6eac085f2edb22e6ecef17152c9ce1ab"},
}
MARKER = ".gausam-data.json"
REPOSITORY = Path(__file__).resolve().parent


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, destination, expected_sha256=None, validator=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination) + ".lock"):
        return _download(url, destination, expected_sha256, validator)


def _download(url, destination, expected_sha256=None, validator=None):
    """Stream to a partial file, validate, then atomically publish; retry transport failures."""
    destination = Path(destination)
    if urlsplit(url).scheme != "https":
        raise ValueError("Automatic downloads require an HTTPS URL")
    destination.parent.mkdir(parents=True, exist_ok=True)
    def validate(path):
        if not path.stat().st_size:
            raise ValueError("Downloaded file is empty")
        if expected_sha256 and sha256(path) != expected_sha256.lower():
            raise ValueError("SHA256 mismatch; the remote file may have changed")
        if validator:
            validator(path)
    if destination.is_file():
        try:
            validate(destination)
            return destination
        except Exception:
            LOG.warning("Cached %s failed validation; fetching a fresh copy", destination.name)
    partial = destination.with_name(destination.name + ".part")
    context = ssl.create_default_context()
    # Add Mozilla roots while retaining system roots (including configured corporate CAs).
    import certifi
    context.load_verify_locations(certifi.where())
    failure = None
    for attempt in range(3):
        try:
            LOG.info("Downloading %s from %s (attempt %d/3)", destination.name, urlsplit(url).hostname, attempt + 1)
            if urlsplit(url).hostname == "drive.google.com":
                import gdown
                result = gdown.download(url=url, output=str(partial), fuzzy=True,
                                        quiet=False, use_cookies=False,
                                        verify=os.environ.get("SSL_CERT_FILE") or True)
                if result is None:
                    raise RuntimeError("Google Drive download failed; check public sharing and download quota")
            else:
                request = Request(url, headers={"User-Agent": "GauSAM/1.0", "Accept-Encoding": "identity"})
                with urlopen(request, timeout=60, context=context) as response, partial.open("wb") as stream:
                    total = int(response.headers.get("Content-Length", "0"))
                    received = 0
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        stream.write(chunk)
                        received += len(chunk)
                    if total and received != total:
                        raise OSError("Incomplete HTTP response")
            validate(partial)
            os.replace(partial, destination)
            return destination
        except Exception as error:
            failure = error
            partial.unlink(missing_ok=True)
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Could not download/validate {destination.name}: {failure}") from failure


def check_zip(path):
    with zipfile.ZipFile(path) as archive:
        if sum(item.file_size for item in archive.infolist()) > 64 * 1024**3:
            raise ValueError("Archive exceeds the 64 GiB extraction limit")
        if not archive.infolist() or archive.testzip() is not None:
            raise ValueError("Invalid or corrupt ZIP archive")


def extract_zip(path, destination):
    """Reject traversal, links and colliding names before extracting any entry."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(path) as archive:
        entries, seen = [], set()
        if sum(item.file_size for item in archive.infolist()) > 64 * 1024**3:
            raise ValueError("Archive exceeds the 64 GiB extraction limit")
        for item in archive.infolist():
            name = PurePosixPath(item.filename.replace("\\", "/"))
            if name.is_absolute() or ".." in name.parts or any(":" in part for part in name.parts):
                raise ValueError(f"Unsafe archive path: {item.filename}")
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError("Archive symlinks are not supported")
            target = destination.joinpath(*name.parts).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Archive path escapes extraction directory")
            if not item.is_dir():
                identity = str(target).casefold()
                if identity in seen:
                    raise ValueError("Duplicate archive filename")
                seen.add(identity)
                entries.append((item, target))
        for item, target in entries:
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item) as source, target.open("wb") as out:
                shutil.copyfileobj(source, out)


def pair_directories(images, masks):
    images, masks = Path(images), Path(masks)
    if not images.is_dir() or not masks.is_dir():
        return []
    im = {p.name: p for p in images.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
    gt = {p.name: p for p in masks.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
    if not im or im.keys() != gt.keys() or any(not p.stat().st_size for p in [*im.values(), *gt.values()]):
        return []
    return [(im[name], gt[name]) for name in sorted(im)]


def pair_files(root, dataset, split):
    root = Path(root)
    pairs = pair_directories(root / split / "images", root / split / "masks")
    if not pairs and dataset == "cvc" and split == "test":
        pairs = pair_directories(root / "PNG" / "Original", root / "PNG" / "Ground Truth")
    return pairs


def dataset_fingerprint(root, dataset, splits):
    root = Path(root)
    files = sorted(p for split in splits for pair in pair_files(root, dataset, split) for p in pair)
    digest = hashlib.sha256()
    for p in files:
        digest.update(p.relative_to(root).as_posix().encode() + b"\0" + sha256(p).encode() + b"\n")
    return digest.hexdigest()


def dataset_ready(root, dataset, splits):
    root = Path(root)
    if not all(pair_files(root, dataset, split) for split in splits):
        return False
    marker = root / MARKER
    if marker.is_file():
        info = json.loads(marker.read_text(encoding="utf-8"))
        if info.get("dataset") != dataset:
            raise ValueError("Prepared dataset type conflicts with --dataset")
        for name, digest in info["files"].items():
            path = (root / name).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("Invalid path in dataset inventory")
            if not path.is_file() or sha256(path) != digest:
                return False
    return True


def prepare_polyp(raw, output):
    """Copy the published train/val/test folders without resplitting or re-encoding."""
    candidates = [p.parent for p in Path(raw).rglob("train")
                  if all(pair_files(p.parent, "polyp", split) for split in ("train", "val", "test"))]
    if len(candidates) != 1:
        raise ValueError("Expected one prepared train/val/test dataset in the archive")
    source = candidates[0]
    if dataset_fingerprint(source, "polyp", ("train", "val", "test")) != DATASETS["polyp"]["inventory"]:
        raise ValueError("Archive does not match the published fixed split and image/mask contents")
    for split in ("train", "val", "test"):
        for pair in pair_files(source, "polyp", split):
            for p in pair:
                target = output / p.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, target)


def prepare_cvc(raw, output):
    """Use all PNG pairs as a held-out cross-domain test set."""
    candidates = [p for p in Path(raw).rglob("PNG")
                  if pair_directories(p / "Original", p / "Ground Truth")]
    if len(candidates) != 1:
        raise ValueError("Expected one CVC PNG/Original + PNG/Ground Truth pair")
    for image, mask in pair_directories(candidates[0] / "Original", candidates[0] / "Ground Truth"):
        for folder, source in (("images", image), ("masks", mask)):
            target = output / "test" / folder / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def ensure_dataset(args):
    root = Path(args.data_path).resolve()
    needed = ("train", "val") if args.mode == "train" else (args.split,)
    spec = DATASETS[args.dataset]
    default_root = REPOSITORY / "data" / spec["directory"]
    if root != default_root.resolve():
        if not dataset_ready(root, args.dataset, needed):
            raise ValueError(f"Custom dataset must be an existing, complete prepared directory: {root}")
        LOG.info("Using prepared custom dataset: %s", root)
        return
    root.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root.parent / f".{root.name}.prepare.lock")):
        if dataset_ready(root, args.dataset, needed):
            if args.dataset == "polyp" and dataset_fingerprint(root, "polyp", ("train", "val", "test")) != spec["inventory"]:
                raise ValueError("Default data differs from the published fixed split. Preserve it under another name; "
                                 "the empty default location will be downloaded automatically.")
            LOG.info("Using existing dataset: %s", root)
            return
        managed = (root / MARKER).is_file()
        if root.exists() and any(root.iterdir()) and not managed:
            raise ValueError(f"Incomplete existing dataset at {root}; refusing to overwrite user data")
        archive = download(spec["url"], Path(args.cache_dir).resolve() / spec["archive"], spec["sha256"], check_zip)
        with tempfile.TemporaryDirectory(prefix=".gausam-prepare-", dir=root.parent) as temporary:
            stage = Path(temporary)
            raw, output = stage / "raw", stage / "ready"
            extract_zip(archive, raw)
            (prepare_polyp if args.dataset == "polyp" else prepare_cvc)(raw, output)
            if not dataset_ready(output, args.dataset, needed):
                raise ValueError("Prepared dataset failed pair validation")
            splits = ("train", "val", "test") if args.dataset == "polyp" else ("test",)
            info = {"dataset": args.dataset, "preparation_version": 2, "source": spec["url"],
                    "archive_sha256": spec["sha256"],
                    "counts": {split: len(pair_files(output, args.dataset, split)) for split in splits},
                    "files": {p.relative_to(output).as_posix(): sha256(p)
                              for split in splits for pair in pair_files(output, args.dataset, split) for p in pair}}
            (output / MARKER).write_text(json.dumps(info, indent=2), encoding="utf-8")
            if managed:
                previous = json.loads((root / MARKER).read_text(encoding="utf-8"))
                if previous.get("files") != info["files"]:
                    raise ValueError("Repair source differs from the existing inventory; preserve the old directory first")
                for name, digest in info["files"].items():
                    target = root / name
                    if target.exists() and sha256(target) != digest:
                        raise ValueError(f"Local data was modified: {target}; refusing to overwrite it")
                for name in info["files"]:
                    target = root / name
                    if not target.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(output / name, target)
            else:
                if root.exists():
                    root.rmdir()
                os.replace(output, root)
        LOG.info("Dataset prepared: %s", root)


def ensure_sam(args):
    import torch
    path = REPOSITORY / "sam_ckpt" / SAM_FILES[args.model_type]
    args.checkpoint = str(path)
    def validate(file):
        weights = torch.load(file, map_location="cpu", weights_only=True)
        key = "image_encoder.patch_embed.proj.weight"
        expected = {"vit_b": 768, "vit_l": 1024, "vit_h": 1280}[args.model_type]
        depth = {"vit_b": 12, "vit_l": 24, "vit_h": 32}[args.model_type]
        required = {key, "image_encoder.pos_embed", f"image_encoder.blocks.{depth - 1}.attn.qkv.weight",
                    "image_encoder.neck.0.weight", "prompt_encoder.no_mask_embed.weight",
                    "prompt_encoder.pe_layer.positional_encoding_gaussian_matrix",
                    "mask_decoder.iou_token.weight", "mask_decoder.mask_tokens.weight"}
        if (not isinstance(weights, dict) or not required <= weights.keys()
                or tuple(weights[key].shape) != (expected, 3, 16, 16)):
            raise ValueError(f"Expected complete original SAM {args.model_type} pretrained weights")
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".prepare.lock"):
        if path.is_file():
            validate(path)
            LOG.info("Using existing SAM weights: %s", path)
            return
        url = "https://dl.fbaipublicfiles.com/segment_anything/" + SAM_FILES[args.model_type]
        download(url, path, validator=validate)


def prepare_inputs(args):
    """Run on each worker before init_process_group; filesystem locks serialize preparation."""
    if not args.data_path:
        args.data_path = str(REPOSITORY / "data" / DATASETS[args.dataset]["directory"])
    args.checkpoint = str(REPOSITORY / "sam_ckpt" / SAM_FILES[args.model_type])
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ensure_dataset(args)
    # A full GauSAM checkpoint already includes SAM; eval/resume need no second download.
    if not args.resume:
        ensure_sam(args)

"""Optional feature-map visualization used by the Gaussian decoder."""
from pathlib import Path
import cv2
import numpy as np


def visualize_embeddings(embeddings, save_dir="embedding_visualizations"):
    """Save each channel of the first [B,C,H,W] feature map as a PNG."""
    destination = Path(save_dir)
    destination.mkdir(parents=True, exist_ok=True)
    channels = embeddings[0].detach().cpu().numpy()
    for index, channel in enumerate(channels):
        low, high = channel.min(), channel.max()
        normalized = ((channel - low) / (high - low) * 255).astype(np.uint8) if high > low else np.zeros_like(channel, dtype=np.uint8)
        path = destination / f"channel_{index:03d}.png"
        if not cv2.imwrite(str(path), normalized):
            raise OSError(f"Could not save {path}")

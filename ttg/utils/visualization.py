from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import torch


def _to_image(tensor: torch.Tensor):
    return ((tensor.detach().cpu().squeeze().numpy() + 1.0) / 2.0).clip(0.0, 1.0)


def visualize_comparison(
    coarse_map: torch.Tensor,
    refined_map: torch.Tensor,
    final_map: torch.Tensor,
    gt_map: torch.Tensor,
    sample_id: int,
    save_dir: str | Path,
) -> Path:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    images = [_to_image(coarse_map), _to_image(refined_map), _to_image(final_map), _to_image(gt_map)]
    titles = ["Coarse Phase", "Refinement Phase", "Final Output", "Ground Truth"]
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    for ax, img, title in zip(axes, images, titles):
        im = ax.imshow(img, cmap="jet", vmin=0, vmax=1, origin="lower")
        ax.set_title(title)
        ax.axis("off")
    fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04).set_label("Normalized RSS")
    fig.tight_layout()
    path = save_dir / f"comparison_sample_{sample_id}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_loss_curves(history: dict, save_path: str | Path) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history.get("train_loss", []), label="Train")
    val = history.get("val_loss")
    if val:
        plt.plot(epochs, val, label="Validation", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

from dataset import get_loaders, MEAN, STD, NUM_CLASSES
from model import UNet

CLASS_COLORS = np.array([
    [255, 111,   0],   # 0 = pet (amber)
    [  0, 105,  92],   # 1 = background (teal)
    [158, 158, 158],   # 2 = border (gray)
], dtype=np.uint8)

CLASS_NAMES = ["Pet", "Background", "Border"]


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std  = torch.tensor(STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def colorize(mask_np: np.ndarray) -> np.ndarray:
    return CLASS_COLORS[mask_np.clip(0, NUM_CLASSES - 1)]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader = get_loaders(root="data", batch_size=16)
    imgs_batch, masks_batch = next(iter(val_loader))

    model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load("best.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(imgs_batch.to(device))        # [B, C, H, W]
        preds  = logits.argmax(dim=1).cpu()          # [B, H, W]

    n      = min(4, len(imgs_batch))
    idxs   = random.sample(range(len(imgs_batch)), n)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(idxs):
        axes[row][0].imshow(denormalize(imgs_batch[idx]))
        axes[row][1].imshow(colorize(masks_batch[idx].numpy()))
        axes[row][2].imshow(colorize(preds[idx].numpy()))
        axes[row][0].set_title("Original")
        axes[row][1].set_title("Ground Truth")
        axes[row][2].set_title("Prediction")
        for ax in axes[row]:
            ax.axis("off")

    legend = [mpatches.Patch(color=np.array(c) / 255, label=n_)
              for c, n_ in zip(CLASS_COLORS, CLASS_NAMES)]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=12)
    plt.tight_layout()
    plt.savefig("results.png", dpi=150, bbox_inches="tight")
    print("Saved results.png")


if __name__ == "__main__":
    main()

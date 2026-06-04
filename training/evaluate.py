import torch
import torchmetrics

from dataset import get_loaders, NUM_CLASSES
from model import UNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader = get_loaders(root="data", batch_size=8)

    model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load("best.pth", map_location=device))
    model.eval()

    iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=NUM_CLASSES).to(device)
    acc = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(device)

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs).argmax(dim=1)   # [B, C, H, W] → [B, H, W]
            iou.update(preds, masks)
            acc.update(preds, masks)

    print(f"IoU (mJaccard) : {iou.compute():.4f}")
    print(f"Pixel Accuracy : {acc.compute():.4f}")


if __name__ == "__main__":
    main()

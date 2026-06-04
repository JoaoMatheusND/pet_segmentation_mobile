import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from dataset import get_loaders
from model import UNet


def validate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            loss = criterion(model(imgs), masks)   # [B,C,H,W] → scalar
            total += loss.item()
    return total / len(loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_loaders(root="data", batch_size=8)

    model     = UNet(in_channels=3, num_classes=3).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    epochs = 20

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{epochs}", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            loss = criterion(model(imgs), masks)   # model output [B,C,H,W]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss    = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        marker = " ✓" if val_loss < best_val_loss else ""
        print(f"Epoch {epoch+1:02d}/{epochs} | train={train_loss:.4f} | val={val_loss:.4f}{marker}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best.pth")

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()

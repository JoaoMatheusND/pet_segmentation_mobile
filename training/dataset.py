import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

IMG_SIZE    = 256
NUM_CLASSES = 3

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

_img_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def _mask_tf(pil_mask: Image.Image) -> torch.Tensor:
    # Convert to 'L' first — trimap PNGs may be palette mode 'P'
    pil_mask = pil_mask.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    t = torch.as_tensor(np.array(pil_mask, dtype=np.int64))
    return (t - 1).clamp(0, NUM_CLASSES - 1)   # {1,2,3} → {0,1,2}


class PetSegDataset(Dataset):
    def __init__(self, root: str, split: str = "trainval", download: bool = True):
        self._base = OxfordIIITPet(
            root=root,
            split=split,
            target_types="segmentation",
            download=download,
        )

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int):
        img, mask = self._base[idx]
        return _img_tf(img), _mask_tf(mask)


def get_loaders(root: str = "data", batch_size: int = 8, num_workers: int = 4):
    full = PetSegDataset(root=root, split="trainval")

    n_train = min(1000, len(full))
    n_val   = min(1200, len(full))

    train_ds = Subset(full, range(n_train))
    val_ds   = Subset(full, range(n_train, n_val))

    pin = torch.cuda.is_available()

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    return train_loader, val_loader

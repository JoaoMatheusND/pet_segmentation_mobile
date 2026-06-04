import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # pad if spatial dims differ (odd input sizes)
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 3):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = Down(64,  128)
        self.enc3 = Down(128, 256)
        self.enc4 = Down(256, 512)
        self.bot  = Down(512, 1024)

        self.dec4 = Up(1024 + 512, 512)
        self.dec3 = Up(512  + 256, 256)
        self.dec2 = Up(256  + 128, 128)
        self.dec1 = Up(128  +  64,  64)

        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b  = self.bot(e4)

        x = self.dec4(b,  e4)
        x = self.dec3(x,  e3)
        x = self.dec2(x,  e2)
        x = self.dec1(x,  e1)
        return self.head(x)               # [B, C, H, W]  NCHW


import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """Two 3D convolution layers with batch normalization and ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    """
    3D U-Net for multi-class brain tumour segmentation.

    Input:  (B, 4, D, H, W)
    Output: (B, 4, D, H, W) raw class logits
    """

    def __init__(self, in_channels=4, num_classes=4, base_channels=16):
        super().__init__()

        self.enc1 = ConvBlock3D(in_channels, base_channels)
        self.enc2 = ConvBlock3D(base_channels, base_channels * 2)
        self.enc3 = ConvBlock3D(base_channels * 2, base_channels * 4)

        self.pool = nn.MaxPool3d(kernel_size=2)

        self.bottleneck = ConvBlock3D(
            base_channels * 4, base_channels * 8
        )

        self.up3 = nn.ConvTranspose3d(
            base_channels * 8, base_channels * 4, kernel_size=2, stride=2
        )
        self.dec3 = ConvBlock3D(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose3d(
            base_channels * 4, base_channels * 2, kernel_size=2, stride=2
        )
        self.dec2 = ConvBlock3D(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose3d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.dec1 = ConvBlock3D(base_channels * 2, base_channels)

        self.output = nn.Conv3d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat((d3, e3), dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat((d2, e2), dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat((d1, e1), dim=1))

        return self.output(d1)


if __name__ == "__main__":
    model = UNet3D()
    sample = torch.randn(1, 4, 96, 96, 96)

    with torch.no_grad():
        prediction = model(sample)

    print("3D U-Net architecture test")
    print("Input shape:", tuple(sample.shape))
    print("Output shape:", tuple(prediction.shape))
    print("Number of classes:", prediction.shape[1])
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    print("Finite output:", bool(torch.isfinite(prediction).all()))

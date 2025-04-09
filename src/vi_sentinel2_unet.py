import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
from pathlib import Path
import rasterio
import matplotlib.pyplot as plt


class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, dropout_p=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout2d(p=dropout_p)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout2d(p=dropout_p)

    def forward(self, x, skip):
        x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.relu1(self.bn1(x))
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.relu2(self.bn2(x))
        x = self.dropout2(x)
        return x


class Sentinel2UNet(nn.Module):
    def __init__(self, num_classes, pretrained=True, backbone="resnet18", dropout_p=0.3):
        super().__init__()

        # Define backbone architecture and channel dimensions
        self.dropout_p = dropout_p
        if backbone == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.encoder = resnet18(weights=weights)
            self.encoder_channels = [64, 64, 128, 256, 512]
        elif backbone == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.encoder = resnet50(weights=weights)
            self.encoder_channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose 'resnet18' or 'resnet50'")

        print(f"Initializing UNet with {backbone} backbone")
        print(f"Encoder channels: {self.encoder_channels}")

        # Replace the first conv layer to accept 13 channels for Sentinel-2
        self.encoder.conv1 = nn.Conv2d(13, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Encoder layers
        self.layer0 = nn.Sequential(
            self.encoder.conv1,
            self.encoder.bn1,
            self.encoder.relu,
            self.encoder.maxpool
        )
        self.layer1 = self.encoder.layer1
        self.layer2 = self.encoder.layer2
        self.layer3 = self.encoder.layer3
        self.layer4 = self.encoder.layer4

        # Define decoder channel dimensions
        decoder_channels = [256, 128, 64, 32]

        self.decoder4 = UNetDecoderBlock(self.encoder_channels[4], self.encoder_channels[3], decoder_channels[0],
                                         dropout_p=self.dropout_p)
        self.decoder3 = UNetDecoderBlock(decoder_channels[0], self.encoder_channels[2], decoder_channels[1],
                                         dropout_p=self.dropout_p)
        self.decoder2 = UNetDecoderBlock(decoder_channels[1], self.encoder_channels[1], decoder_channels[2],
                                         dropout_p=self.dropout_p)
        self.decoder1 = UNetDecoderBlock(decoder_channels[2], self.encoder_channels[0], decoder_channels[3],
                                         dropout_p=self.dropout_p)

        # Final upsampling to go from H/4 to full resolution
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(decoder_channels[3], 32, kernel_size=4, stride=4, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=self.dropout_p),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=self.dropout_p)
        )

        self.final_dropout = nn.Dropout2d(p=self.dropout_p)
        # Final classification layer
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # Save input shape for possible final resizing
        input_shape = x.shape

        # Encoder forward pass
        x0 = self.layer0(x)  # H/4
        x1 = self.layer1(x0)  # H/4
        x2 = self.layer2(x1)  # H/8
        x3 = self.layer3(x2)  # H/16
        x4 = self.layer4(x3)  # H/32

        # Decoder with skip connections
        d4 = self.decoder4(x4, x3)
        d3 = self.decoder3(d4, x2)
        d2 = self.decoder2(d3, x1)
        d1 = self.decoder1(d2, x0)

        # Final upsampling to restore original resolution
        d1_upsampled = self.final_upsample(d1)  # Upsample from H/4 to H
        out = self.final_dropout(d1_upsampled)
        out = self.final_conv(out)

        # Ensure output matches input spatial dimensions exactly
        if out.shape[2:] != input_shape[2:]:
            out = nn.functional.interpolate(out, size=input_shape[2:], mode="bilinear", align_corners=True)

        return out


if __name__ == "__main__":
    # Test both backbone architectures
    for backbone in ["resnet18", "resnet50"]:
        print(f"\nTesting {backbone} backbone...")
        num_classes = 8
        model = Sentinel2UNet(num_classes=num_classes, backbone=backbone)
        model.eval()

        patch_id = "100"
        patch_dir = Path("data/patch_dataset")
        img_path = patch_dir / "images" / f"image_{patch_id}.tif"
        lbl_path = patch_dir / "labels" / f"label_{patch_id}.tif"

        with rasterio.open(img_path) as src:
            img = src.read().astype(np.float32) / 10000.0
            transform = src.transform
        input_tensor = torch.tensor(img).unsqueeze(0)

        with rasterio.open(lbl_path) as src:
            label = src.read(1)

        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
            print(f"Output shape: {output.shape}")
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            print(f"Predicted class map shape: {pred.shape}")
            print(f"Unique predicted classes: {np.unique(pred)}")

        # Show results for this backbone
        rgb = img[[3, 2, 1], :, :]
        rgb = np.clip(rgb / rgb.max(), 0, 1).transpose(1, 2, 0)

        plt.figure(figsize=(15, 6))
        plt.suptitle(f"Results with {backbone} backbone")
        plt.subplot(1, 3, 1)
        plt.title("Input Image (RGB)")
        plt.imshow(rgb)
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.title("Ground Truth")
        plt.imshow(label, cmap="tab20", interpolation="none")
        plt.axis("off")
        plt.subplot(1, 3, 3)
        plt.title("Model Prediction")
        plt.imshow(pred, cmap="tab20", interpolation="none")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
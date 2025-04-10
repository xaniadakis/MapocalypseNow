import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        return self.block(x)


class UNetResNet(nn.Module):
    def __init__(self, encoder_depth=50, num_classes=10, pretrained=True):
        super().__init__()

        # Load pre-trained ResNet as encoder
        if encoder_depth == 18:
            self.encoder = models.resnet18(pretrained=pretrained)
            encoder_channels = [64, 64, 128, 256, 512]
        elif encoder_depth == 50:
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.encoder = models.resnet50(weights=weights)
            encoder_channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported encoder depth: {encoder_depth}")

        # Modify first conv layer to accept 13-channel input
        original_first_conv = self.encoder.conv1
        self.encoder.conv1 = nn.Conv2d(
            13, original_first_conv.out_channels,
            kernel_size=original_first_conv.kernel_size,
            stride=original_first_conv.stride,
            padding=original_first_conv.padding,
            bias=original_first_conv.bias
        )

        # Initialize weights for new first layer
        if pretrained:
            with torch.no_grad():
                # Average the weights of the original 3 channels across new 13 channels
                new_weight = original_first_conv.weight.data.mean(dim=1, keepdim=True).expand(-1, 13, -1, -1)
                self.encoder.conv1.weight.data = new_weight

        # Remove the original fully connected layer
        self.encoder.fc = nn.Identity()

        # Encoder blocks
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu),
            nn.Sequential(self.encoder.maxpool, self.encoder.layer1),
            self.encoder.layer2,
            self.encoder.layer3,
            self.encoder.layer4
        ])

        # Decoder blocks
        decoder_channels = [256, 128, 64, 32]
        self.decoder_layers = nn.ModuleList()

        # Create decoder layers with skip connections
        in_ch = encoder_channels[-1]  # Start with the last encoder output
        for i in range(len(decoder_channels)):
            skip_ch = encoder_channels[-(i + 2)]
            out_ch = decoder_channels[i]
            self.decoder_layers.append(
                DecoderBlock(in_ch + skip_ch, out_ch)
            )
            in_ch = out_ch  # Next decoder layer takes output of previous as input

            # Final convolution
            # self.final_conv = nn.Sequential(
            #     nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1),
            #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # )
            self.final_conv = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)

            # Store channels for skip connections
            self.encoder_channels = encoder_channels

    def forward(self, x):
        # Encoder
        skip_connections = []
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            if i < len(self.encoder_layers) - 1:  # Don't store last layer as skip
                skip_connections.append(x)

        # Decoder with skip connections (using at least 2 as required)
        for i, decoder in enumerate(self.decoder_layers):
            skip = skip_connections[-i - 1]  # Get corresponding skip connection
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat([x, skip], dim=1))

        x = self.final_conv(x)
        return x
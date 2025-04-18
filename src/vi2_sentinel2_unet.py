import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights, ResNet101_Weights, ResNet152_Weights, ResNet18_Weights


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, preds, targets):
        ce_loss = F.cross_entropy(preds, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets, mask):
        preds = torch.softmax(preds, dim=1)
        num_classes = preds.shape[1]

        dice = 0.0
        for cls in range(1, num_classes):  # skip background if index 0
            pred_flat = preds[:, cls].contiguous().view(-1)
            target_flat = (targets == cls).float().view(-1)
            mask_flat = mask.view(-1)

            pred_flat = pred_flat[mask_flat == 1]
            target_flat = target_flat[mask_flat == 1]

            intersection = (pred_flat * target_flat).sum()
            union = pred_flat.sum() + target_flat.sum()
            dice += (2. * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice / (num_classes - 1)

class AdaptiveLoss(nn.Module):
    def __init__(self, class_weights, num_classes, init_ce=0.2, init_dice=0.5, init_focal=0.3):
        super().__init__()
        self.raw_weights = nn.Parameter(torch.tensor([init_ce, init_dice, init_focal]))  # now all together

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0, weight=class_weights)
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()

        self.num_classes = num_classes
        self.normalized_weights = None

        assert class_weights is None or len(class_weights) == num_classes, \
            f"Length of class_weights ({len(class_weights)}) != num_classes ({num_classes})"

    def forward(self, outputs, labels, mask):
        ce = self.ce_loss(outputs, labels)
        dice = self.dice_loss(outputs, labels, mask)
        focal = self.focal_loss(outputs, labels)

        # use softplus instead of relu for smoother gradient flow
        weights = F.softplus(self.raw_weights)
        weights = weights / (weights.sum() + 1e-6)

        self.normalized_weights = weights.detach()

        return weights[0] * ce + weights[1] * dice + weights[2] * focal


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.atrous_block1 = nn.Conv2d(in_channels, out_channels, 1, padding=0, dilation=1)
        self.atrous_block6 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.ReLU()
        )
        self.conv1 = nn.Conv2d(out_channels * 5, out_channels, 1)

    def forward(self, x):
        size = x.shape[2:]

        out1 = self.atrous_block1(x)
        out2 = self.atrous_block6(x)
        out3 = self.atrous_block12(x)
        out4 = self.atrous_block18(x)
        out5 = self.global_pool(x)
        out5 = F.interpolate(out5, size=size, mode='bilinear', align_corners=True)

        x = torch.cat([out1, out2, out3, out4, out5], dim=1)
        return self.conv1(x)


class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=8):
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(channels // reduction_ratio, channels)
        )
        self.sigmoid = nn.Sigmoid()

        # Spatial attention
        # TODO: check if this conv is better
        # self.conv = nn.utils.spectral_norm(
        #     nn.Conv2d(2, 1, kernel_size=7, padding=3)
        # )
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))
        channel_out = self.sigmoid(avg_out + max_out).unsqueeze(-1).unsqueeze(-1)
        x = x * channel_out

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_out = torch.cat([avg_out, max_out], dim=1)
        spatial_out = self.sigmoid(self.conv(spatial_out))
        return x * spatial_out

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn = self.global_pool(x)
        attn = self.fc(attn)
        return x * attn

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p):
        super().__init__()

        # self.attn = AttentionBlock(in_channels)
        self.attn = CBAM(in_channels)  # Using CBAM instead of simple attention

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        # Residual connection
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        ) if in_channels != out_channels else nn.Identity()


    def forward(self, x):
        x = self.attn(x)
        return self.block(x) + self.shortcut(x)

class UNetResNet(nn.Module):
    def __init__(self, encoder_depth=50, num_classes=10, pretrained=True, dropout_p=0.2):
        super().__init__()

        # Load pre-trained ResNet as encoder
        if encoder_depth == 18:
            self.encoder = models.resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
            encoder_channels = [64, 64, 128, 256, 512]
        elif encoder_depth == 50:
            self.encoder = models.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
            encoder_channels = [64, 256, 512, 1024, 2048]
        elif encoder_depth == 101:
            self.encoder = models.resnet101(weights=ResNet101_Weights.DEFAULT if pretrained else None)
            encoder_channels = [64, 256, 512, 1024, 2048]
        elif encoder_depth == 152:
            self.encoder = models.resnet152(weights=ResNet152_Weights.DEFAULT if pretrained else None)
            encoder_channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported encoder depth: {encoder_depth}")

        self.encoder.layer4[0].conv1.stride = (1, 1)
        self.encoder.layer4[0].conv2.stride = (1, 1)
        self.encoder.layer4[0].downsample[0].stride = (1, 1)

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

        self.aspp = ASPP(in_channels=encoder_channels[-1], out_channels=encoder_channels[-1])

        # # Freeze first half of the encoder (conv1, bn1, layer1, layer2)
        # freeze_layers = [self.encoder.conv1, self.encoder.bn1, self.encoder.layer1, self.encoder.layer2]
        # for layer in freeze_layers:
        #     for param in layer.parameters():
        #         param.requires_grad = False

        # Decoder blocks
        decoder_channels = [256, 128, 64, 32]
        self.decoder_layers = nn.ModuleList()

        # Create decoder layers with skip connections
        in_ch = encoder_channels[-1]  # Start with the last encoder output
        for i in range(len(decoder_channels)):
            skip_ch = encoder_channels[-(i + 2)]
            out_ch = decoder_channels[i]
            self.decoder_layers.append(
                DecoderBlock(in_ch + skip_ch, out_ch, dropout_p=dropout_p)
            )
            in_ch = out_ch  # Next decoder layer takes output of previous as input

            # Store channels for skip connections
            self.encoder_channels = encoder_channels

        self.final_conv = nn.Sequential(
            nn.Dropout2d(p=dropout_p),  # new
            nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)
        )

    def forward(self, x):
        # Encoder
        skip_connections = []
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            if i < len(self.encoder_layers) - 1:  # Don't store last layer as skip
                skip_connections.append(x)
        x = self.aspp(x)

        # Decoder with skip connections (using at least 2 as required)
        for i, decoder in enumerate(self.decoder_layers):
            skip = skip_connections[-i - 1]  # Get corresponding skip connection
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat([x, skip], dim=1))

        x = self.final_conv(x)
        return x
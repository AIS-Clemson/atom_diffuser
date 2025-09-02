# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 03:32:18 2024

@author: MaxGr
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffUNetV3(nn.Module):
    def __init__(self, in_channels=1, base=8, time_embed_dim=128):
        super().__init__()

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # Shared encoder block
        def make_encoder_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU()
            )

        self.enc1 = make_encoder_block(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = make_encoder_block(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = make_encoder_block(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = make_encoder_block(base * 4, base * 8)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck fusion
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base * 16, base * 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base * 16, base * 16, 3, padding=1), nn.ReLU()
        )

        # Decoder
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = make_encoder_block(base * 24, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = make_encoder_block(base * 12, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = make_encoder_block(base * 6, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = make_encoder_block(base * 3, base)

        # Output heads
        self.out_theta = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base * 16, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3)
        )
        
        self.out_lambda = nn.Sequential(
            nn.Conv2d(base, base, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(base, base, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(base, 1, 1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x0, xT, t):
        B = x0.size(0)
        t_embed = self.time_embed(t.view(B, 1).float() / 100.0)
        t_broadcast = t_embed.view(B, -1, 1, 1)

        # Shared encoder
        e0_1 = self.enc1(x0)
        e0_2 = self.enc2(self.pool1(e0_1))
        e0_3 = self.enc3(self.pool2(e0_2))
        e0_4 = self.enc4(self.pool3(e0_3))

        eT_1 = self.enc1(xT)
        eT_2 = self.enc2(self.pool1(eT_1))
        eT_3 = self.enc3(self.pool2(eT_2))
        eT_4 = self.enc4(self.pool3(eT_3))
        
        # Bottleneck fusion
        # b = self.bottleneck(torch.cat([self.pool3(e0_3), self.pool3(eT_3)], dim=1)) + t_broadcast
        b = self.bottleneck(torch.cat([self.pool4(e0_4), self.pool4(eT_4)], dim=1)) + t_broadcast

        # Decoder with dual skip
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e0_4, eT_4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e0_3, eT_3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e0_2, eT_2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e0_1, eT_1], dim=1))

        theta_params = self.out_theta(b)
        lambda_map = self.out_lambda(d1)

        return theta_params, lambda_map


class DiffUNet(nn.Module):
    def __init__(self, in_channels=2, base=32, time_embed_dim=64):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base * 2, base * 2, 3, padding=1), nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base * 4, base * 4, 3, padding=1), nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base * 4, base * 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base * 8, base * 8, 3, padding=1), nn.ReLU()
        )

        # Decoder
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(base * 8, base * 4, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base * 4, base * 4, 3, padding=1), nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base * 4, base * 2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base * 2, base * 2, 3, padding=1), nn.ReLU()
        )

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base * 2, base, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU()
        )

        # Output heads
        self.out_lambda = nn.Sequential(
            nn.Conv2d(base, 1, 3, padding=1),
            nn.Sigmoid()  # pixel-wise decay map
        )

        self.out_theta = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),           # [B, base, 1, 1]
            nn.Flatten(),                      # [B, base]
            nn.Linear(base, 64),
            nn.ReLU(),
            nn.Linear(64, 3)                   # trans parameters
        )

    def forward(self, x0, xT, t):
        x = torch.cat([x0, xT], dim=1)  # [B, 2, H, W]
        B, _, H, W = x.shape

        # time embedding
        t_embed = self.time_embed(t.view(B, 1).float() / 1000.0)  # [B, C]
        t_broadcast = t_embed.view(B, -1, 1, 1)

        # inject t into encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3)) + t_broadcast  # inject t here

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        lambda_map = self.out_lambda(d1)        # [B, 1, H, W]
        theta_params = self.out_theta(d1)       # [B, 6] → [B, 2, 3]

        return theta_params, lambda_map


class DiffUNetEX(nn.Module):
    def __init__(self, in_channels=1, base=32, time_embed_dim=256):
        super().__init__()

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # Shared encoder block
        def make_encoder_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU()
            )

        self.enc1 = make_encoder_block(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = make_encoder_block(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = make_encoder_block(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck fusion
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base * 8, base * 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base * 8, base * 8, 3, padding=1), nn.ReLU()
        )

        # Decoder
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = make_encoder_block(base * 12, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = make_encoder_block(base * 6, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = make_encoder_block(base * 3, base)

        # Output heads
        self.out_lambda = nn.Sequential(
            nn.Conv2d(base, 1, 3, padding=1),
            nn.Sigmoid()
        )

        self.out_theta = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3)
        )

    def forward(self, x0, xT, t):
        B = x0.size(0)
        t_embed = self.time_embed(t.view(B, 1).float() / 100.0)
        t_broadcast = t_embed.view(B, -1, 1, 1)

        # Shared encoder
        e0_1 = self.enc1(x0)
        e0_2 = self.enc2(self.pool1(e0_1))
        e0_3 = self.enc3(self.pool2(e0_2))

        eT_1 = self.enc1(xT)
        eT_2 = self.enc2(self.pool1(eT_1))
        eT_3 = self.enc3(self.pool2(eT_2))

        # Bottleneck fusion
        b = self.bottleneck(torch.cat([self.pool3(e0_3), self.pool3(eT_3)], dim=1)) + t_broadcast

        # Decoder with dual skip
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e0_3, eT_3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e0_2, eT_2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e0_1, eT_1], dim=1))

        lambda_map = self.out_lambda(d1)
        theta_params = self.out_theta(b)

        return theta_params, lambda_map


# model = DiffUNet().to(device)
# model = DiffUNetEX().to(device)

# # Instantiate the model
# model = AE()

# # Test the model with dummy inputs
# img1 = torch.randn(1, 2, 512, 512)  # Batch of 4 RGB images
# img2 = torch.randn(1, 1, 512, 512)
# output = model(img1)

# print(f"Output shape: {output.shape}")  # Expected: [4, 1, 512, 512]





class DiffUNetV2(nn.Module):
    def __init__(self, in_channels=1, base=32, time_embed_dim=256):
        super().__init__()

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # Shared encoder block
        def make_encoder_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU()
            )

        self.enc1 = make_encoder_block(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = make_encoder_block(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = make_encoder_block(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck fusion
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base * 8, base * 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base * 8, base * 8, 3, padding=1), nn.ReLU()
        )

        # Decoder
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = make_encoder_block(base * 12, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = make_encoder_block(base * 6, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = make_encoder_block(base * 3, base)

        # Output heads
        self.out_theta = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3)
        )
        
        self.out_lambda = nn.Sequential(
            nn.Conv2d(base, base, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(base, base, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(base, 1, 1, stride=1, padding=0),
            nn.Sigmoid()
        )
        

    def forward(self, x0, xT, t):
        B = x0.size(0)
        t_embed = self.time_embed(t.view(B, 1).float() / 100.0)
        t_broadcast = t_embed.view(B, -1, 1, 1)

        # Shared encoder
        e0_1 = self.enc1(x0)
        e0_2 = self.enc2(self.pool1(e0_1))
        e0_3 = self.enc3(self.pool2(e0_2))

        eT_1 = self.enc1(xT)
        eT_2 = self.enc2(self.pool1(eT_1))
        eT_3 = self.enc3(self.pool2(eT_2))

        # Bottleneck fusion
        b = self.bottleneck(torch.cat([self.pool3(e0_3), self.pool3(eT_3)], dim=1)) + t_broadcast

        # Decoder with dual skip
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e0_3, eT_3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e0_2, eT_2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e0_1, eT_1], dim=1))

        theta_params = self.out_theta(b)
        lambda_map = self.out_lambda(d1)

        return theta_params, lambda_map







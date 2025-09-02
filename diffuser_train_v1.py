# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 21:45:38 2025

@author: MaxGr
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"

import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms

from torchsummary import summary
# from models import SiameseUNet, SiameseVGG, AE, UNet, UNet_L3, UNet_L3_dual, UNet_L3_v3, UNet_Shift

print('torch.version: ',torch. __version__)
print('torch.version.cuda: ',torch.version.cuda)
print('torch.cuda.is_available: ',torch.cuda.is_available())
print('torch.cuda.device_count: ',torch.cuda.device_count())
print('torch.cuda.current_device: ',torch.cuda.current_device())
device_default = torch.cuda.current_device()
torch.cuda.device(device_default)
# device = torch.device("cuda")
# print(torch.cuda.get_arch_list())
device = "cuda" if torch.cuda.is_available() else "cpu"
print('torch.cuda.get_device_name: ',torch.cuda.get_device_name(device_default))



# =========================
# Models
# =========================
import torch
import torch.nn as nn
import torch.nn.functional as F

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


import torch
import torch.nn as nn
import torch.nn.functional as F

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
model = DiffUNetEX().to(device)


# =========================
# Dataset
# =========================
# Define transforms with normalization
transform = transforms.Compose([
    transforms.ToPILImage(),                              # Convert to PIL image
    # transforms.Resize((256, 256)),                       # Resize to match input dimensions
    transforms.ToTensor(),                               # Convert to tensor [0, 1]
    # transforms.Normalize(mean=[0.5], std=[0.5]),         # Normalize: mean 0.5, std 0.5 (example values)
])

mask_transform = transforms.Compose([
    transforms.ToPILImage(),                              # Convert to PIL image
    # transforms.Resize((256, 256)),                       # Resize to match input dimensions
    transforms.ToTensor(),                               # Convert to tensor [0, 1]
    # transforms.Normalize(mean=[0.5], std=[0.5]),         # Normalize: mean 0.5, std 0.5 (example values)
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),                              # Convert to PIL image
    # transforms.Resize((512, 512)),                       # Resize to match input dimensions
    transforms.ToTensor(),                               # Convert to tensor [0, 1]
    # transforms.Normalize(mean=[0.5], std=[0.5]),         # Normalize: mean 0.5, std 0.5 (example values)
])



data_folder = 'D://Data/TEM-ImageNet-v1.3-master/TEM-ImageNet-v1.3-master/'

# image_dir = data_folder+'/image/'
image_dir = data_folder+'/noBackgroundnoNoise/'
# mask_dir = data_folder+'/gaussianMask/'
# mask_dir = data_folder+'/smallcircularMask/'


class CustomDataloading(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = sorted(os.listdir(self.image_dir))
        self.transform = transform
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert("L")        
        image = np.array(image)
        # if self.transform:
        image = self.transform(image)
        return image

# =========================
# Utility Functions
# =========================
def norm(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)
    return x

def torch_norm(x):
    x = (x - torch.min(x)) / (torch.max(x) - torch.min(x) + 1e-8)
    return x

def img2uint8(x):
    x = (norm(x) * 255).astype(np.uint8)
    return x

def generate_theta(batch_size, img_size, 
                   angle_range=(-5, 5), 
                   scale_range=(1.0, 1.0), 
                   translate_range=(-0.3, 0.3)):  # ratio
    angles = (torch.empty(batch_size).uniform_(*angle_range)*np.pi/180).to(device)
    scales = torch.empty(batch_size).uniform_(*scale_range).to(device)
    tx_rel = torch.empty(batch_size).uniform_(*translate_range).to(device)
    ty_rel = torch.empty(batch_size).uniform_(*translate_range).to(device)

    tx = tx_rel #* img_size
    ty = ty_rel #* img_size

    thetas = []
    for i in range(batch_size):
        c, s = torch.cos(angles[i]), torch.sin(angles[i])
        matrix = torch.tensor([
            [scales[i] * c, -scales[i] * s, tx[i]],
            [scales[i] * s,  scales[i] * c, ty[i]]
        ]).to(device)
        thetas.append(matrix)
    return torch.stack(thetas)  # shape: [B, 2, 3]

def warp_image_with_flow(img, flow, mode='zeros'):
    B, C, H, W = img.shape
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=img.device),
        torch.linspace(-1, 1, W, device=img.device),
        indexing='ij'
    )
    grid = torch.stack((x, y), dim=2).unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 2]

    norm_flow = torch.zeros_like(flow)
    norm_flow[:, 0, :, :] = flow[:, 0, :, :] * 2 / (W - 1)  # dx
    norm_flow[:, 1, :, :] = flow[:, 1, :, :] * 2 / (H - 1)  # dy

    warped_grid = grid + norm_flow.permute(0, 2, 3, 1)  # [B, H, W, 2]
    return F.grid_sample(img, warped_grid, align_corners=True, padding_mode=mode)

def warp_image_with_theta(img, theta, mode='zeros'):
    grid = F.affine_grid(theta, size=img.size(), align_corners=True)
    return F.grid_sample(img, grid, align_corners=True, padding_mode=mode)


def affine_theta_to_flow(theta, size):
    B, _, H, W = size
    device = theta.device

    grid_base = F.affine_grid(torch.eye(2, 3).unsqueeze(0).repeat(B, 1, 1).to(device), size=size, align_corners=True)
    grid_theta = F.affine_grid(theta, size=size, align_corners=True)

    norm_flow = grid_theta - grid_base  # [B, H, W, 2], range [-2,2]

    # Convert to pixel scale
    flow_x = norm_flow[..., 0] * (W - 1) / 2
    flow_y = norm_flow[..., 1] * (H - 1) / 2
    flow = torch.stack((flow_x, flow_y), dim=1)  # [B, 2, H, W]
    return flow, norm_flow.permute(0, 3, 1, 2)

def add_poisson_noise(image, scale=1.0):
    image = image.clamp(min=0)
    noisy_image = torch.poisson(image * scale) / scale
    return noisy_image

def add_scan_noise(image, jitter_std=1.0):
    B, C, H, W = image.shape
    shifts = torch.normal(mean=0.0, std=jitter_std, size=(B, H, 1), device=image.device).round().long()  # [B, H, 1]
    col_indices = torch.arange(W, device=image.device).view(1, 1, W).expand(B, H, W)  # [B, H, W]
    shifted_cols = (col_indices + shifts) % W  # [B, H, W]
    shifted_cols = shifted_cols.unsqueeze(1).expand(B, C, H, W)  # [B, C, H, W]
    noisy_image = torch.gather(image, dim=3, index=shifted_cols)
    return noisy_image

def params_to_theta(theta_param):
    # B = theta_param.shape[0]
    theta = theta_param[:, 0]
    tx = theta_param[:, 1]
    ty = theta_param[:, 2]
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    row1 = torch.stack([cos, -sin, tx], dim=1)
    row2 = torch.stack([sin,  cos, ty], dim=1)
    theta_matrix = torch.stack([row1, row2], dim=1)  # [B, 2, 3]
    return theta_matrix

def decay_function(x0, noise, t, T, threshold=0.5):
    b, c, h, w = x0.shape
    alpha = t.view(-1, 1, 1, 1) / T
    noise_up = F.interpolate(noise, size=(h, w), mode='bicubic', align_corners=False)
    decay_factor = torch.clamp((noise_up - threshold) * alpha * 2, 0, 1)  # [B,1,H,W]
    xt = x0 * (1 - decay_factor)
    return xt

def interpolate_affine_matrix(A_target, t, T, device='cuda'):
    B = A_target.shape[0]

    if isinstance(t, (int, float)):
        alpha = torch.full((B, 1, 1), float(t) / T, device=device)
    elif isinstance(t, torch.Tensor):
        if t.ndim == 1:
            alpha = (t / T).view(B, 1, 1)  # [B, 1, 1]
        elif t.ndim >= 2:
            alpha = (t / T).view(B, 1, 1)  # from [B, 1, 1, 1] or so
        else:
            raise ValueError(f"Unexpected shape for t: {t.shape}")
    else:
        raise TypeError("t must be int, float, or torch.Tensor")

    identity = torch.tensor([[1., 0., 0.],
                             [0., 1., 0.]], device=device).unsqueeze(0).expand(B, -1, -1)

    A_t = identity + alpha * (A_target - identity)  # elementwise broadcast
    return A_t  # [B, 2, 3]



# =========================
# Test
# =========================
def file_to_tensor(file):
    img_RGB = cv2.imread(test_dir+file)
    img = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2GRAY) 
    img_tensor = test_transform(img).unsqueeze(0).to(device)
    return img_tensor 

def draw_flow_arrows(flow_map, mask, flow_magnitudes, arrow_color=(0, 0, 255)):
    scale = 1
    canvas = flow_magnitudes.copy()
    mask_bin = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        dx = flow_map[0, cy, cx]
        dy = flow_map[1, cy, cx]
        tip_x = int(cx + dx * scale)
        tip_y = int(cy + dy * scale)
        cv2.arrowedLine(canvas, (cx, cy), (tip_x, tip_y), arrow_color, thickness=1, tipLength=0.3)
    return canvas

def draw_flow_arrows_full(flow_map, flow_magnitudes, arrow_color=(0, 0, 255), num=100):
    scale = 1
    _, H, W = flow_map.shape
    canvas = flow_magnitudes.copy()
    spacing = H//num
    for y in range(0, H, spacing):
        for x in range(0, W, spacing):
            dx = flow_map[0, y, x]
            dy = flow_map[1, y, x]
            tip_x = int(x + dx * scale)
            tip_y = int(y + dy * scale)
            cv2.arrowedLine(canvas, (tip_x, tip_y), (x, y),  arrow_color, thickness=1, tipLength=0.3)
    return canvas

def eval_images(x0, xt, xT, t, model, batch=8, scale_factor=None):
    hgrid = []
    model.eval()
    with torch.no_grad():
        pred_params, pred_lamda = model(x0, xT, t)
        pred_theta = params_to_theta(pred_params)        # [B, 2, 3]

        xt_pred = x0*(pred_lamda)
        pred_theta_t = interpolate_affine_matrix(pred_theta, t, T, device='cuda')
        xt_pred = warp_image_with_theta(xt_pred, pred_theta_t, mode='zeros')
           
    flow_maps, _ = affine_theta_to_flow(pred_theta, x0.size())
    # flow_maps = pred_flow
    flow_magnitudes = torch.norm(flow_maps, dim=1, keepdim=True)

    for i in range(len(x0)):
        if scale_factor:
            if scale_factor <= 1:
                x0 = F.interpolate(x0, scale_factor=0.5, mode='bilinear', align_corners=True)
                xt = F.interpolate(xt, scale_factor=0.5, mode='bilinear', align_corners=True)
                xT = F.interpolate(xT, scale_factor=0.5, mode='bilinear', align_corners=True)
                flow_maps = F.interpolate(flow_maps, scale_factor=0.5, mode='bilinear', align_corners=True)
                xt_pred = F.interpolate(xt_pred, scale_factor=0.5, mode='bilinear', align_corners=True)
                flow_magnitudes = F.interpolate(flow_magnitudes, scale_factor=0.5, mode='bilinear', align_corners=True)
                pred_lamda = F.interpolate(pred_lamda, scale_factor=0.5, mode='bilinear', align_corners=True)

            else:
                x0 = F.interpolate(x0, size=(scale_factor, scale_factor), mode='bilinear', align_corners=True)
                xt = F.interpolate(xt, size=(scale_factor, scale_factor), mode='bilinear', align_corners=True)
                xT = F.interpolate(xT, size=(scale_factor, scale_factor), mode='bilinear', align_corners=True)
                flow_maps = F.interpolate(flow_maps, size=(scale_factor, scale_factor), mode='bilinear', align_corners=True)
                xt_pred = F.interpolate(xt_pred, size=(scale_factor, scale_factor), mode='bilinear', align_corners=True)
                flow_magnitudes = F.interpolate(flow_magnitudes, size=(scale_factor, scale_factor), mode='bilinear', align_corners=True)
                pred_lamda = F.interpolate(pred_lamda, size=(scale_factor, scale_factor), mode='bilinear', align_corners=True)

        img0 = x0[i,0,:,:].detach().cpu().numpy()
        imgt = xt[i,0,:,:].detach().cpu().numpy()
        imgT = xT[i,0,:,:].detach().cpu().numpy()
        # flow_map = norm(flow_maps[i,0,:,:].detach().cpu().numpy())
        imgt_pred = xt_pred[i,0,:,:].detach().cpu().numpy()
        flow_magnitude = flow_magnitudes[i,0,:,:].detach().cpu().numpy()
        lamda = pred_lamda[i,0,:,:].detach().cpu().numpy()


        img0_RGB = cv2.cvtColor(img2uint8(img0), cv2.COLOR_GRAY2RGB)
        imgt_RGB = cv2.cvtColor(img2uint8(imgt), cv2.COLOR_GRAY2RGB)
        imgT_RGB = cv2.cvtColor(img2uint8(imgT), cv2.COLOR_GRAY2RGB)
        imgt_pred_R = cv2.cvtColor(img2uint8(imgt_pred), cv2.COLOR_GRAY2RGB)
        flow_magnitude = cv2.cvtColor(img2uint8(flow_magnitude), cv2.COLOR_GRAY2RGB)
        # flow_magnitude = cv2.applyColorMap(img2uint8(flow_magnitude), cv2.COLORMAP_JET)
        lamda = cv2.cvtColor(img2uint8(lamda), cv2.COLOR_GRAY2RGB)

        # img_t_B[:,:,1:] = 0
        imgt_pred_R[:,:,:2] = 0
        match_map = cv2.addWeighted(imgt_RGB, 1, imgt_pred_R, 1, 1)
        
        # flow_arrows = draw_flow_arrows(flow_maps[i,:,:,:].detach().cpu().numpy(), img, flow_magnitude)
        flow_arrows = draw_flow_arrows_full(flow_maps[i,:,:,:].detach().cpu().numpy(), flow_magnitude, num=10)

        vgrid = np.vstack((img0_RGB, imgt_RGB, imgT_RGB, flow_arrows, imgt_pred_R, match_map))
        hgrid.append(vgrid)
    grid = np.hstack((hgrid[:batch]))
    return grid

# grid = eval_images(test_masks, test_masks_t, model, 8)
# save_image = img2uint8(grid)
# cv2.imwrite(f'./{output_dir}/{epoch+1}.png', save_image)


# =========================
# Entry
# =========================
from transformers import get_cosine_schedule_with_warmup
torch.set_printoptions(sci_mode=False)

num_epochs = 50
batch_size = 32
learning_rate = 1e-4
checkpoint_dir = "weights"
output_dir = './exp/output'
test_dir = './samples/'

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

dataset = CustomDataloading(image_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

# Initialize the model, optimizer, and loss
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

criterion_1 = nn.L1Loss().to(device)
criterion_2 = nn.MSELoss().to(device)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100, 
    num_training_steps=len(dataloader)*num_epochs
)
# plot_lr_scheduler(optimizer, scheduler, num_epochs=num_epochs)


T=100
lr_list = []
loss_list = []
model.to(device)
best_loss = float("inf")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    loop = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch [{epoch+1}/{num_epochs}]")
    for batch_idx, (imgs) in loop:
        x0 = imgs.to(device)
        
        # -------- Generate affine grid --------
        theta = generate_theta(batch_size=x0.size(0), img_size=x0.size(2))  # [B, 2, 3]
        # grid = F.affine_grid(theta, imgs.size(), align_corners=True)
        
        # -------- Generate perlin noise --------
        shot_scale = (10 + 2 * torch.randn(1)).to(device)
        scan_std = float(random.randint(1, 5))
        res = random.choice([2, 3, 4, 5, 6, 7, 8])
        threshold = random.uniform(0.4, 0.6)

        noise = torch.rand_like(x0)[:, :, :res, :res]  # perlin noise
        background = torch.ones(x0.shape, device=device)*torch.rand(()) # background noise

        # -------- Decay --------
        # x0 = decay_function(x0, noise, 0, T)
        # theta_0 = interpolate_affine_matrix(theta, t, T, device='cuda')
        # x0 = warp_image_with_theta(x0, theta_0, mode='zeros')
        x0_n = x0 + background
        x0_n = add_poisson_noise(x0_n, scale=shot_scale)
        x0_n = add_scan_noise(x0_n, jitter_std=scan_std)
        
        T0 = torch.tensor([T], dtype=torch.float32, device=device).expand(x0.size(0))
        xT = decay_function(x0, noise, T0, T)
        # theta_T = interpolate_affine_matrix(theta, T, T, device='cuda')
        xT_trans = warp_image_with_theta(xT, theta, mode='zeros')
        xT_n = xT_trans + background
        xT_n = add_poisson_noise(xT_n, scale=shot_scale)
        xT_n = add_scan_noise(xT_n, jitter_std=scan_std)
        
        t = torch.randint(1, T-1, (x0.size(0),), dtype=torch.float32, device=device)
        xt = decay_function(x0, noise, t, T)
        theta_t = interpolate_affine_matrix(theta, t, T, device='cuda')
        xt_gt = warp_image_with_theta(xt, theta_t, mode='zeros')
        # xt_n = xt_trans + background
        # xt_n = add_poisson_noise(xt_n, scale=shot_scale)
        # xt_n = add_scan_noise(xt_n, jitter_std=scan_std)
        
        pred_params, pred_lamda = model(x0_n, xT_n, t)
        pred_theta = params_to_theta(pred_params)        # [B, 2, 3]

        xt_pred = x0*(pred_lamda)
        pred_theta_t = interpolate_affine_matrix(pred_theta, t, T, device='cuda')
        xt_pred = warp_image_with_theta(xt_pred, pred_theta_t, mode='zeros')
            
        loss = criterion_1(xt_pred, xt_gt)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        running_loss += loss.item()
        avg_loss = running_loss / (batch_idx + 1)
        loop.set_postfix({"loss": loss.item(), "avg_loss": avg_loss})
        
    epoch_loss = running_loss / len(dataloader)
    current_lr = optimizer.param_groups[0]['lr']
    lr_list.append(current_lr)
    loss_list.append(epoch_loss)
    # if epoch>50: scheduler.step()
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} Current Lr: {current_lr}")
    # print(f'loss_rec: {loss_rec:.4f} | loss_dice: {loss_dice:.4f} | loss_tv: {loss_tv:.4f}')
    # print(f'loss_rec: {loss_rec:.4f} | loss_flow: {loss_flow:.4f}')
    # print(f'loss_rec: {loss_rec:.4f} | loss_flow: {loss_smooth:.4f}')
    
    # Save the model checkpoint if the loss improved
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"best_model.pth"))
        print("Saved Best Model!")
    
    # # Save checkpoint every epoch
    # torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth"))
    
    # =========================
    # Validation
    # =========================
    test_indices = torch.randperm(len(x0))[:8]
    test_x0 = x0_n[test_indices].to(device)    # [8, C, H, W]
    test_xt = xt_gt[test_indices].to(device)    # [8, C, H, W]
    test_xT = xT_n[test_indices].to(device)    # [8, C, H, W]
    test_t = t[test_indices].to(device)    # [8, C, H, W]

    grid = eval_images(test_x0, test_xt, test_xT, test_t, model, 8)
    save_image = img2uint8(grid)
    cv2.imwrite(f'./{output_dir}/{epoch+1}.png', save_image)
    
    # =========================
    # Test & save
    # =========================
    file_1 = '6MX JEOL ADF1  0325.png'
    file_2 = '6MX JEOL ADF1  0326.png'
    file_3 = '6MX JEOL ADF1  0327.png'
    
    file_4 = 'tilt_JEOL_1.png'
    file_5 = 'tilt_JEOL_2.png'
    file_6 = 'tilt_JEOL_3.png'

    test_t = torch.tensor([50.0], dtype=torch.float32, device=device)
    imgs0, imgs1, imgs2 = file_to_tensor(file_1), file_to_tensor(file_2), file_to_tensor(file_3)
    grid1 = eval_images(imgs0, imgs1, imgs2, test_t, model, 2, 512)  
    imgs0, imgs1, imgs2 = file_to_tensor(file_4), file_to_tensor(file_5), file_to_tensor(file_6)
    grid2 = eval_images(imgs0, imgs1, imgs2, test_t, model, 2, 512)
    grid = np.hstack((grid1, grid2))
    save_image = img2uint8(grid)
    cv2.imwrite(f'./{output_dir}/{epoch+1}_test.png', save_image)
    
    print("Training Complete")
    plt.figure()
    plt.plot(loss_list)
    plt.savefig('loss_plot.png')
    # plt.show()
    



# =========================
# Test
# =========================
import imageio.v2 as imageio

file_1 = 'tilt_JEOL_1.png'
file_2 = 'tilt_JEOL_2.png'
file_3 = 'tilt_JEOL_3.png'

file_4 = 'tilt_JEOL_1_output_gaussianMask+.png'
file_5 = 'tilt_JEOL_2_output_gaussianMask+.png'
file_6 = 'tilt_JEOL_3_output_gaussianMask+.png'

imgs0, imgs1, imgs2 = file_to_tensor(file_1), file_to_tensor(file_2), file_to_tensor(file_3)
masks0, masks1, masks2 = file_to_tensor(file_4), file_to_tensor(file_5), file_to_tensor(file_6)

model.load_state_dict(torch.load(f'{checkpoint_dir}/DiffUNet_best_model.pth'))
model.eval()
hgrid = []
frames = []

T = 100
num_grid = 10
# crop_size = 800
for i in tqdm(range(T), total=T):
    # if i%(T//num_grid)!=0: continue

    # crop0, crop1, crop2 = random_crop_triplet(imgs0, imgs1, imgs2, crop_size=crop_size)
    t = torch.tensor([i], dtype=torch.float32, device=device).expand(imgs0.size(0))
    with torch.no_grad():
        pred_params, pred_lamda = model(imgs0, imgs1, t)
        pred_theta = params_to_theta(pred_params)
        # xt_raw = imgs0 * pred_lamda
        theta_t = interpolate_affine_matrix(pred_theta, t, T, device=imgs0.device)
        xt_pred = warp_image_with_theta(imgs0, theta_t, mode='zeros')
        lamda_warp = warp_image_with_theta(pred_lamda, theta_t, mode='zeros')

    flow_maps, _ = affine_theta_to_flow(pred_theta, imgs0.size())
    flow_magnitudes = torch.norm(flow_maps, dim=1, keepdim=True)

    img0 = imgs0[0,0,:,:].detach().cpu().numpy()
    img1 = imgs1[0,0,:,:].detach().cpu().numpy()
    flow_map = norm(flow_maps[0,0,:,:].detach().cpu().numpy())
    pred_img1 = xt_pred[0,0,:,:].detach().cpu().numpy()
    flow_magnitude = flow_magnitudes[0,0,:,:].detach().cpu().numpy()
    lamda = pred_lamda[0,0,:,:].detach().cpu().numpy()
    lamda1 = lamda_warp[0,0,:,:].detach().cpu().numpy()

    img0_RGB = cv2.cvtColor(img2uint8(img0), cv2.COLOR_GRAY2RGB)
    img1_RGB = cv2.cvtColor(img2uint8(img1), cv2.COLOR_GRAY2RGB)
    pred_img1_R = cv2.cvtColor(img2uint8(pred_img1), cv2.COLOR_GRAY2RGB)
    pred_img1_RGB = cv2.cvtColor(img2uint8(pred_img1), cv2.COLOR_GRAY2RGB)
    flow_magnitude = cv2.cvtColor(img2uint8(flow_magnitude), cv2.COLOR_GRAY2RGB)
    lamda = cv2.applyColorMap(img2uint8(255-lamda), cv2.COLORMAP_JET)
    lamda1 = cv2.applyColorMap(img2uint8(255-lamda1), cv2.COLORMAP_JET)

    # img_t_B[:,:,1:] = 0
    pred_img1_R[:,:,:2] = 0
    match_map = cv2.addWeighted(img1_RGB, 1, pred_img1_R, 1, 1)
    
    # flow_arrows = draw_flow_arrows(flow_maps[i,:,:,:].detach().cpu().numpy(), img, flow_magnitude)
    flow_arrows = draw_flow_arrows_full(flow_maps[0,:,:,:].detach().cpu().numpy(), flow_magnitude, num=20)

    if T%(i+1)==0: 
        vgrid = np.vstack((img0_RGB, lamda, img1_RGB, match_map, flow_arrows))
        vgrid = cv2.resize(vgrid, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        hgrid.append(vgrid)
    
    overlay = cv2.addWeighted(lamda1, 0.2, match_map, 1, 1)
    frame = np.hstack((lamda, overlay, match_map))
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)
    

grid = np.hstack((hgrid))
file_name = f'{file_1[:-4]}_{T}'
cv2.imwrite(f'./test/grid_{file_name}.png', grid)
imageio.mimsave(f'./test/frames_{file_name}.mp4', frames, fps=10)
# imageio.mimsave(f'./test/frames_{file_name}.mp4', frames, fps=10, format='ffmpeg')
cv2.imwrite(f'./test/lamda_{file_name}.png', lamda)
cv2.imwrite(f'./test/match_map_{file_name}.png', match_map)
cv2.imwrite(f'./test/flow_arrows_{file_name}.png', flow_arrows)






# =========================
# Test
# =========================

# -------- Generate affine grid --------
theta = generate_theta(batch_size=x0.size(0), img_size=x0.size(2))  # [B, 2, 3]

# -------- Generate perlin noise --------
shot_scale = (10 + 2 * torch.randn(1)).to(device)
scan_std = float(random.randint(1, 5))
res = random.choice([2, 3, 4, 5, 6, 7, 8])
threshold = random.uniform(0.4, 0.6)

noise = torch.rand_like(x0)[:, :, :res, :res]  # perlin noise
background = torch.ones(x0.shape, device=device)*torch.rand(()) # background noise

# -------- Decay --------
# x0 = decay_function(x0, noise, 0, T)
# theta_0 = interpolate_affine_matrix(theta, t, T, device='cuda')
# x0 = warp_image_with_theta(x0, theta_0, mode='zeros')
x0_n = x0 + background
x0_n = add_poisson_noise(x0_n, scale=shot_scale)
x0_n = add_scan_noise(x0_n, jitter_std=scan_std)

T0 = torch.tensor([T], dtype=torch.float32, device=device).expand(x0.size(0))
xT = decay_function(x0, noise, T0, T, threshold=0)
# theta_T = interpolate_affine_matrix(theta, T, T, device='cuda')
xT_trans = warp_image_with_theta(xT, theta, mode='zeros')
xT_n = xT_trans + background
xT_n = add_poisson_noise(xT_n, scale=shot_scale)
xT_n = add_scan_noise(xT_n, jitter_std=scan_std)

decay_list = []
for i in tqdm(range(T), total=T):
    if (i+1)%(T//num_grid)!=0: continue
    # t = torch.randint(1, T-1, (x0.size(0),), dtype=torch.float32, device=device)
    t = torch.tensor([i], dtype=torch.float32, device=device).expand(x0.size(0))
    xt = decay_function(x0, noise, t, T, threshold=0)
    theta_t = interpolate_affine_matrix(theta, t, T, device='cuda')
    xt_gt = warp_image_with_theta(xt, theta_t, mode='zeros')
    xt_n = xt_gt + background
    xt_n = add_poisson_noise(xt_n, scale=shot_scale)
    xt_n = add_scan_noise(xt_n, jitter_std=scan_std)
    decay_list.append(xt_n)

decay_cat = torch.cat(decay_list, dim=-1)

hgrid = []
for i in range(8):
    img0 = x0_n[i,0,:,:].detach().cpu().numpy()
    img1 = xT_n[i,0,:,:].detach().cpu().numpy()
    decay = decay_cat[i,0,:,:].detach().cpu().numpy()
    
    img0_RGB = cv2.cvtColor(img2uint8(img0), cv2.COLOR_GRAY2RGB)
    img1_RGB = cv2.cvtColor(img2uint8(img1), cv2.COLOR_GRAY2RGB)
    decay_RGB = cv2.cvtColor(img2uint8(decay), cv2.COLOR_GRAY2RGB)

    hgrid.append(np.hstack((img0_RGB, decay_RGB, img1_RGB)))

grid = np.vstack((hgrid))
cv2.imwrite(f'./exp/test_grid.png', grid)




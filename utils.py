# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 21:45:38 2025

@author: hao9
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

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Test
# =========================
def file_to_tensor(file, transform):
    img_RGB = cv2.imread(file)
    img = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2GRAY) 
    img_tensor = transform(img).unsqueeze(0).to(device)
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





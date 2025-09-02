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

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
from torchsummary import summary

device_default = torch.cuda.current_device()
torch.cuda.device(device_default)
device = "cuda" if torch.cuda.is_available() else "cpu"

print('torch.version: ',torch. __version__)
print('torch.version.cuda: ',torch.version.cuda)
print('torch.cuda.is_available: ',torch.cuda.is_available())
print('torch.cuda.device_count: ',torch.cuda.device_count())
print('torch.cuda.current_device: ',torch.cuda.current_device())
print(torch.cuda.get_arch_list())
print('torch.cuda.get_device_name: ',torch.cuda.get_device_name(device_default))


# =========================
# Dataset
# =========================
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
# Test
# =========================
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
        lamda = cv2.cvtColor(img2uint8(lamda), cv2.COLOR_GRAY2RGB)

        # img_t_B[:,:,1:] = 0
        imgt_pred_R[:,:,:2] = 0
        match_map = cv2.addWeighted(imgt_RGB, 1, imgt_pred_R, 1, 1)
        flow_arrows = draw_flow_arrows_full(flow_maps[i,:,:,:].detach().cpu().numpy(), flow_magnitude, num=10)

        vgrid = np.vstack((img0_RGB, imgt_RGB, imgT_RGB, flow_arrows, imgt_pred_R, match_map))
        hgrid.append(vgrid)
    grid = np.hstack((hgrid[:batch]))
    return grid


# =========================
# Entry
# =========================
from transformers import get_cosine_schedule_with_warmup
torch.set_printoptions(sci_mode=False)

from models import DiffUNetEX
from utils import *
model = DiffUNetEX().to(device)
transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),])
data_folder = 'D://Data/TEM-ImageNet-v1.3-master/TEM-ImageNet-v1.3-master/'
image_dir = data_folder+'/noBackgroundnoNoise/'
# image_dir = data_folder+'/image/'
# mask_dir = data_folder+'/gaussianMask/'
# mask_dir = data_folder+'/smallcircularMask/'

num_epochs = 500
batch_size = 32
learning_rate = 1e-4
checkpoint_dir = "./weights/"
output_dir = './exp/output/'
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

# =========================
# Train
# =========================
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
            
        # -------- Loss --------
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
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} Current Lr: {current_lr}")

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
    imgs0, imgs1, imgs2 = file_to_tensor(test_dir+file_1, transform), file_to_tensor(test_dir+file_2, transform), file_to_tensor(test_dir+file_3, transform)
    grid1 = eval_images(imgs0, imgs1, imgs2, test_t, model, 2, 512)  
    imgs0, imgs1, imgs2 = file_to_tensor(test_dir+file_1, transform), file_to_tensor(test_dir+file_2, transform), file_to_tensor(test_dir+file_3, transform)
    grid2 = eval_images(imgs0, imgs1, imgs2, test_t, model, 2, 512)
    
    grid = np.hstack((grid1, grid2))
    save_image = img2uint8(grid)
    cv2.imwrite(f'./{output_dir}/{epoch+1}_test.png', save_image)
    
    print("Training Complete")
    plt.figure()
    plt.plot(loss_list)
    plt.savefig('loss_plot.png')
    # plt.show()
    

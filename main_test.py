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
# Entry
# =========================
from models import DiffUNetEX
from utils import *
model = DiffUNetEX().to(device)
transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),])
checkpoint_dir = "./weights/"
test_dir = './samples/'

# =========================
# Test
# =========================
import imageio.v2 as imageio

file_1 = 'tilt_JEOL_1.png'
file_2 = 'tilt_JEOL_2.png'
file_3 = 'tilt_JEOL_3.png'

imgs0, imgs1, imgs2 = file_to_tensor(test_dir+file_1, transform), file_to_tensor(test_dir+file_2, transform), file_to_tensor(test_dir+file_3, transform)

model.load_state_dict(torch.load(f'{checkpoint_dir}/DiffUNet_best_model.pth'))
model.eval()
hgrid = []
frames = []

T = 100
num_grid = 10
for i in tqdm(range(T), total=T):
    # if i%(T//num_grid)!=0: continue
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
    flow_arrows = draw_flow_arrows_full(flow_maps[0,:,:,:].detach().cpu().numpy(), flow_magnitude, num=20)

    if T%(i+1)==0: 
        vgrid = np.vstack((img0_RGB, lamda, img1_RGB, match_map, flow_arrows))
        vgrid = cv2.resize(vgrid, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
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
cv2.imwrite(f'./test/lamda_{file_name}.png', lamda)
cv2.imwrite(f'./test/match_map_{file_name}.png', match_map)
cv2.imwrite(f'./test/flow_arrows_{file_name}.png', flow_arrows)




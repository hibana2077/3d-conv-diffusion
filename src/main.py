import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torch.optim import AdamW
from datetime import datetime
import json

from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models import SinusoidalPosEmb, ConvBlock, Denoiser, Diffusion
from timm.models.ghostnet import ghostnet_050
from tridd_models import TriDD
from utils import add_noise_process, correct_subset

# -------------------------
# 模型與資料設定參數
# -------------------------
dataset_path = '~/datasets'
cuda = True
DEVICE = torch.device("cuda:0" if cuda else "cpu")

datasets_dict = {
    "MNIST": (28, 28, 1),
    "CIFAR10": (32, 32, 3),
    "CelebA": (64, 64, 3),
}
dataset = 'CIFAR10'
img_size = datasets_dict[dataset]

train_batch_size = 512
inference_batch_size = 512
test_batch_size = 10
lr = 5e-4
epochs = 40
gt_weight = 3

label_dim = 10
triddm_proj = 32
classes = 10

seed = 114514
torch.manual_seed(seed)
np.random.seed(seed)

transform = transforms.Compose([
    transforms.ToTensor(),
])

kwargs = {'num_workers': 1, 'pin_memory': True}

# -------------------------
# 載入判別器並固定參數
# -------------------------
discriminator = ghostnet_050(num_classes=classes, in_chans=3 if dataset == 'CIFAR10' else 1)
disc_pth = './cifar10_model.pth' if dataset == 'CIFAR10' else './mnist_model.pth'
discriminator.load_state_dict(torch.load(disc_pth, weights_only=True))
for param in discriminator.parameters():
    param.requires_grad = False
discriminator.to(DEVICE)

# -------------------------
# 載入資料集並進行篩選
# -------------------------
if dataset == 'CIFAR10':
    train_dataset = CIFAR10(dataset_path, transform=transform, train=True, download=True)
    test_dataset  = CIFAR10(dataset_path, transform=transform, train=False, download=True)
else:
    train_dataset = MNIST(dataset_path, transform=transform, train=True, download=True)
    test_dataset  = MNIST(dataset_path, transform=transform, train=False, download=True)

train_dataset = correct_subset(discriminator, train_dataset, DEVICE)
test_dataset  = correct_subset(discriminator, test_dataset, DEVICE)

train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, **kwargs)
test_loader  = DataLoader(dataset=test_dataset,  batch_size=inference_batch_size, shuffle=False, **kwargs)

# -------------------------
# 初始化 TriDD 模型與最佳化器
# -------------------------
model = TriDD(
    img_res=img_size,
    label_dim=label_dim,
    proj=triddm_proj,
).to(DEVICE)

optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)

# 定義損失函數
label_sim_loss = nn.CrossEntropyLoss()
denoising_loss = nn.MSELoss()
l1_loss = nn.L1Loss()

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Device: {DEVICE}")
print(f"Model parameters(M): {count_parameters(model) / 1e6:.2f}")
print(f"Discriminator parameters(M): {count_parameters(discriminator) / 1e6:.2f}")

# -------------------------
# 定義訓練、驗證與生成影像的函數
# -------------------------
def train_epoch(model, discriminator, loader, optimizer, device, epoch, noise_img):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (x, y) in tqdm(enumerate(loader), total=len(loader), desc=f"訓練 Epoch {epoch+1}"):
        optimizer.zero_grad()
        y = y.to(device)
        noise_stack = add_noise_process(x)
        
        # 僅在第一個 epoch 儲存噪聲圖片
        if epoch == 0:
            noise_img["t1000"].append(noise_stack[0, 0])
            noise_img["t500"].append(noise_stack[0, 1])
            noise_img["t10"].append(noise_stack[0, 2])
            noise_img["t0"].append(noise_stack[0, 3])
        
        # 取得不同時間步驟的噪聲
        noise = noise_stack[:, 0].to(device)          # t=1000
        noise_hid_1 = noise_stack[:, 1].to(device)      # t=500
        noise_hid_2 = noise_stack[:, 2].to(device)      # t=10
        original = noise_stack[:, 3].to(device)         # t=0

        out_hids, out = model(noise, y)
        
        # 僅在第一個 epoch儲存生成的影像 (第一張)
        if epoch == 0 and len(noise_img["out"]) < 1:
            noise_img["out"].append(out[0].detach().cpu())
        
        # 計算去噪損失
        deno_loss = (denoising_loss(out_hids[0], noise_hid_1) +
                     denoising_loss(out_hids[1], noise_hid_2) +
                     gt_weight * denoising_loss(out, original))
        
        # 判別器預測 (此處 label_loss 可視需求加權)
        pred = discriminator(out)
        label_loss = label_sim_loss(pred, y)
        
        loss = deno_loss  # 若需要可調整加入 0.3*label_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 計算準確率 (以判別器輸出作為預測)
        pred_labels = torch.argmax(discriminator(out), dim=1)
        total_correct += (pred_labels == y).sum().item()
        total_samples += x.size(0)
    
    avg_loss = total_loss / (batch_idx + 1)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def validate(model, discriminator, loader, device, epoch):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (x, y) in tqdm(enumerate(loader), total=len(loader), desc=f"驗證 Epoch {epoch+1}"):
            y = y.to(device)
            noise_stack = add_noise_process(x)
            noise = noise_stack[:, 0].to(device)
            noise_hid_1 = noise_stack[:, 1].to(device)
            noise_hid_2 = noise_stack[:, 2].to(device)
            original = noise_stack[:, 3].to(device)
            
            out_hids, out = model(noise, y)
            
            deno_loss = (denoising_loss(out_hids[0], noise_hid_1) +
                         denoising_loss(out_hids[1], noise_hid_2) +
                         gt_weight * denoising_loss(out, original))
            
            pred = discriminator(out)
            label_loss = label_sim_loss(pred, y)
            loss = deno_loss  # 若需要可調整加入 0.3*label_loss
            
            total_loss += loss.item()
            pred_labels = torch.argmax(discriminator(out), dim=1)
            total_correct += (pred_labels == y).sum().item()
            total_samples += x.size(0)
    
    avg_loss = total_loss / (batch_idx + 1)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def generate_and_save_images(model, device, test_batch_size, img_size, classes):
    model.eval()
    with torch.no_grad():
        class_labels = torch.arange(0, classes).long().to(device)
        # 產生隨機噪聲 (調整 shape 後餵入模型)
        noise = torch.randn(test_batch_size, *img_size).permute(0, 3, 1, 2).to(device)
        _, out = model(noise, class_labels)
        out = out.cpu()
        nrow = int(math.sqrt(test_batch_size))
        grid = make_grid(out, nrow=nrow, normalize=False)
        save_image(grid, 'generated_images.png')
        print("已儲存生成影像至 generated_images.png")

def save_noise_images(noise_img):
    for key, images in noise_img.items():
        # 將儲存的 list 轉為 tensor 並生成 grid
        stacked = torch.stack(images)
        nrow = int(math.sqrt(stacked.shape[0]))
        grid = make_grid(stacked, nrow=nrow, normalize=False)
        save_image(grid, f'noise_{key}.png')
        print(f"已儲存 {key} 的噪聲影像至 noise_{key}.png")

# -------------------------
# 主訓練流程
# -------------------------
print("Starting training TriDD ...")
model.train()

# 用來儲存各 epoch 訓練與驗證資訊
rec_loss = []
rec_acc = []
rec_test_loss = []
rec_test_acc = []
# 儲存噪聲影像 (僅儲存第一個 epoch的部分結果)
noise_img = {
    "t1000": [],
    "t500": [],
    "t10": [],
    "t0": [],
    "out": [],
}

best_test_acc = 0.0
for epoch in range(epochs):
    train_loss, train_acc = train_epoch(model, discriminator, train_loader, optimizer, DEVICE, epoch, noise_img)
    rec_loss.append(train_loss)
    rec_acc.append(train_acc)
    print(f"Epoch {epoch+1}/{epochs} 訓練完成！平均損失：{train_loss:.4f}，訓練準確率：{train_acc:.4f}")
    
    test_loss, test_acc = validate(model, discriminator, test_loader, DEVICE, epoch)
    rec_test_loss.append(test_loss)
    rec_test_acc.append(test_acc)
    print(f"Epoch {epoch+1}/{epochs} 驗證完成！平均損失：{test_loss:.4f}，驗證準確率：{test_acc:.4f}")
    
    # 儲存最佳模型
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), "TriDD_best_model.pth")
        print("最佳模型已儲存！")

print("訓練結束！")

# 載入最佳模型
model.load_state_dict(torch.load("TriDD_best_model.pth"))

# 儲存實驗參數與結果
exp_data = {
    'rec_loss': rec_loss,
    'rec_acc': rec_acc,
    'dataset': dataset,
    'img_size': img_size,
    'train_batch_size': train_batch_size,
    'inference_batch_size': inference_batch_size,
    'lr': lr,
    'epochs': epochs,
    'seed': seed,
    'label_dim': label_dim,
    'triddm_proj': triddm_proj,
    'classes': classes,
    'gt_weight': gt_weight,
}

file_name = f"rec/exp_{datetime.now().strftime('%Y_%m_%d_%H_%M')}.json"
with open(file_name, 'w') as f:
    json.dump(exp_data, f)
print(f"實驗參數已儲存至 {file_name}")

# 生成影像
generate_and_save_images(model, DEVICE, test_batch_size, img_size, classes)

# 儲存噪聲影像
save_noise_images(noise_img)
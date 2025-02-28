import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import numpy as np

from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torch.optim import Adam, AdamW
from datetime import datetime

import math

from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from models import SinusoidalPosEmb, ConvBlock, Denoiser, Diffusion
from timm.models.ghostnet import ghostnet_050
from tridd_models import TriDD
from utils import add_noise_process, correct_subset

# Model Hyperparameters

dataset_path = '~/datasets'

cuda = True
DEVICE = torch.device("cuda:0" if cuda else "cpu")

dataset = 'CIFAR10'
img_size = (32, 32, 3)   if dataset == "CIFAR10" else (28, 28, 1) # (width, height, channels)

train_batch_size = 4096
inference_batch_size = 1024
test_batch_size = 10
lr = 3e-4
epochs = 200
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

discriminator = ghostnet_050(num_classes=classes, in_chans=3 if dataset == 'CIFAR10' else 1)
disc_pth = './cifar10_model.pth' if dataset == 'CIFAR10' else './mnist_model.pth'
discriminator.load_state_dict(torch.load(disc_pth, weights_only=True))
for param in discriminator.parameters():
    param.requires_grad = False
discriminator.to(DEVICE)

if dataset == 'CIFAR10':
    train_dataset = CIFAR10(dataset_path, transform=transform, train=True, download=True)
    test_dataset  = CIFAR10(dataset_path, transform=transform, train=False, download=True)
    train_dataset = correct_subset(discriminator, train_dataset, DEVICE)
    test_dataset  = correct_subset(discriminator, test_dataset, DEVICE)
else:
    train_dataset = MNIST(dataset_path, transform=transform, train=True, download=True)
    test_dataset  = MNIST(dataset_path, transform=transform, train=False, download=True)
    train_dataset = correct_subset(discriminator, train_dataset, DEVICE)
    test_dataset  = correct_subset(discriminator, test_dataset, DEVICE)

train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, **kwargs)
test_loader  = DataLoader(dataset=test_dataset,  batch_size=inference_batch_size, shuffle=False,  **kwargs)

model = TriDD(
    img_res=img_size,
    label_dim=label_dim,
    proj=triddm_proj,
).to(DEVICE)

# optimizer = Adam(model.parameters(), lr=lr)
# optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
label_sim_loss = nn.CrossEntropyLoss()
denoising_loss = nn.MSELoss()

def count_parameters(model:nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Device: {DEVICE}")
print("Number of model parameters(M): ", count_parameters(model)/1e+6)
print("Number of discriminator parameters(M): ", count_parameters(discriminator)/1e+6)

print("Start training TriDD...")
model.train()

rec_loss = []
rec_acc = []
rec_test_loss = []
rec_test_acc = []
noise_img = {
    "t1000": [],
    "t500": [],
    "t10": [],
    "t0": [],
    "out": [],
}
add_cnt = []
for epoch in range(epochs):
    noise_prediction_loss = 0
    acc = 0
    test_noise_prediction_loss = 0
    test_acc = 0

    # train
    model.train()
    total_correct = 0
    total_samples = 0
    for batch_idx, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()

        y = y.to(DEVICE)
        noise_stack = add_noise_process(x)
        if epoch not in add_cnt:
            noise_img["t1000"].append(noise_stack[0, 0])
            noise_img["t500"].append(noise_stack[0, 1])
            noise_img["t10"].append(noise_stack[0, 2])
            noise_img["t0"].append(noise_stack[0, 3])
        noise = noise_stack[:, 0].to(DEVICE)# t=1000
        noise_hid_1 = noise_stack[:, 1].to(DEVICE)# t=500
        noise_hid_2 = noise_stack[:, 2].to(DEVICE)# t=10
        original = noise_stack[:, 3].to(DEVICE)# t=0
        out_hids, out = model(noise, y)# generated image
        if epoch not in add_cnt:
            # print(f"Save epoch {epoch} noise images")
            noise_img["out"].append(out[0])
            add_cnt.append(epoch)
        # denoising loss
        # deno_loss = denoising_loss(out_hid, noise_hid) + gt_weight * denoising_loss(out, original)
        deno_loss = denoising_loss(out_hids[0], noise_hid_1) + denoising_loss(out_hids[1], noise_hid_2) + gt_weight * denoising_loss(out, original)

        # discriminator loss
        pred = discriminator(out)
        label_loss = label_sim_loss(pred, y)

        loss = deno_loss #+ 0.3*label_loss if epoch>0 and rec_acc[-1] > 0.79 else deno_loss
        loss.backward()
        optimizer.step()

        noise_prediction_loss += loss.item()

        # calculate accuracy
        pred = torch.argmax(discriminator(out), dim=1)
        total_correct += (pred == y).sum().item()
        total_samples += x.size(0)

    acc = total_correct / total_samples
    rec_acc.append(acc)
    rec_loss.append(noise_prediction_loss / batch_idx)
    print(f"Epoch {epoch+1}/{epochs} complete! Denoising Loss: {noise_prediction_loss / batch_idx:.4f}, Accuracy: {acc:.4f}")

    # test
    model.eval()
    total_correct = 0
    total_samples = 0
    for batch_idx, (x, y) in tqdm(enumerate(test_loader), total=len(test_loader)):
        y = y.to(DEVICE)
        noise_stack = add_noise_process(x)
        noise = noise_stack[:, 0].to(DEVICE)
        noise_hid_1 = noise_stack[:, 1].to(DEVICE)
        noise_hid_2 = noise_stack[:, 2].to(DEVICE)
        original = noise_stack[:, 3].to(DEVICE)
        out_hids, out = model(noise, y)
        # denoising loss
        # deno_loss = denoising_loss(out_hid, noise_hid) + gt_weight * denoising_loss(out, original)
        deno_loss = denoising_loss(out_hids[0], noise_hid_1) + denoising_loss(out_hids[1], noise_hid_2) + gt_weight * denoising_loss(out, original)
        # discriminator loss
        pred = discriminator(out)
        label_loss = label_sim_loss(pred, y)
        loss = deno_loss #+ 0.3*label_loss
        test_noise_prediction_loss += loss.item()
        # calculate accuracy
        pred = torch.argmax(discriminator(out), dim=1)
        total_correct += (pred == y).sum().item()
        total_samples += x.size(0)

    test_acc = total_correct / total_samples
    if len(rec_test_acc) == 0 or test_acc > max(rec_test_acc):
        torch.save(model.state_dict(), f"TriDD_best_model.pth")
        print("Best Model saved!")
    rec_test_acc.append(test_acc)
    rec_test_loss.append(test_noise_prediction_loss / batch_idx)
    print(f"Test complete! Test Denoising Loss: {test_noise_prediction_loss / batch_idx:.4f}, Test Accuracy: {test_acc:.4f}")

print("Finish!!")

# load the best model
model.load_state_dict(torch.load("TriDD_best_model.pth"))

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

# save the experiment data in json
import json
file_name = f"rec/exp_{datetime.now().strftime('%Y_%m_%d_%H_%M')}.json"
with open(file_name, 'w') as f:
    json.dump(exp_data, f)

# generate some images (makegrid) (label min(10,classes))
model.eval()
with torch.no_grad():
    # class_labels = torch.randint(0, classes, (test_batch_size,)).to(DEVICE)
    class_labels = torch.arange(0, classes).long().to(DEVICE)
    noise = torch.randn(test_batch_size, *img_size).permute(0, 3, 1, 2).to(DEVICE)
    # noise = noise / noise.std(dim=(1, 2, 3), keepdim=True)
    noise = noise.detach()
    print(f"noise shape: {noise.shape}, label shape: {class_labels.shape}")
    _,out = model(noise, class_labels)
    out = out.cpu()
    print(out.shape)
    nrow = int(math.sqrt(test_batch_size))
    out = make_grid(out, nrow=nrow, normalize=False)
    # save the image
    save_image(out, 'generated_images.png')

# save the noise images
for key in noise_img.keys():
    noise_img[key] = torch.stack(noise_img[key])
    print(f"noise_img[{key}].shape: {noise_img[key].shape}")
    nrow = int(math.sqrt(noise_img[key].shape[0]))
    out = make_grid(noise_img[key], nrow=nrow, normalize=False)
    save_image(out, f'noise_{key}.png')
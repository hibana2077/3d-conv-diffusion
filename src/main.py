import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import numpy as np

from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torch.optim import Adam
from datetime import datetime

import math

from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models import SinusoidalPosEmb, ConvBlock, Denoiser, Diffusion
from timm.models.ghostnet import ghostnet_050
from tridd_models import TriDD

# Model Hyperparameters

dataset_path = '~/datasets'

cuda = True
DEVICE = torch.device("cuda:0" if cuda else "cpu")

dataset = 'MNIST'
img_size = (32, 32, 3)   if dataset == "CIFAR10" else (28, 28, 1) # (width, height, channels)

train_batch_size = 128
inference_batch_size = 64
test_batch_size = 10
lr = 5e-5
epochs = 25

label_dim = 10
triddm_proj = 10
classes = 10

seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)

transform = transforms.Compose([
        transforms.ToTensor(),
])

kwargs = {'num_workers': 1, 'pin_memory': True}

if dataset == 'CIFAR10':
    train_dataset = CIFAR10(dataset_path, transform=transform, train=True, download=True)
    test_dataset  = CIFAR10(dataset_path, transform=transform, train=False, download=True)
else:
    train_dataset = MNIST(dataset_path, transform=transform, train=True, download=True)
    test_dataset  = MNIST(dataset_path, transform=transform, train=False, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, **kwargs)
test_loader  = DataLoader(dataset=test_dataset,  batch_size=inference_batch_size, shuffle=False,  **kwargs)

discriminator = ghostnet_050(num_classes=classes, in_chans=3 if dataset == 'CIFAR10' else 1)
disc_pth = './cifar10_model.pth' if dataset == 'CIFAR10' else './mnist_model.pth'
discriminator.load_state_dict(torch.load(disc_pth, weights_only=True))
for param in discriminator.parameters():
    param.requires_grad = False
discriminator.to(DEVICE)

model = TriDD(
    img_res=img_size,
    label_dim=label_dim,
    proj=triddm_proj,
).to(DEVICE)

optimizer = Adam(model.parameters(), lr=lr)
label_sim_loss = nn.CrossEntropyLoss()
denoising_loss = nn.MSELoss()

def count_parameters(model:nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Device: {DEVICE}")
print("Number of model parameters(M): ", count_parameters(model)/1e+6)

print("Start training TriDD...")
model.train()

rec_loss = []
rec_acc = []
for epoch in range(epochs):
    noise_prediction_loss = 0
    acc = 0
    for batch_idx, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()

        y = y.to(DEVICE)
        noise = torch.randn_like(x).to(DEVICE)
        noise = noise / noise.std(dim=(1, 2, 3), keepdim=True)
        noise = noise.detach()
        out = model(noise, y)# generated image

        # discriminator loss
        pred = discriminator(out)
        loss = label_sim_loss(pred, y)
        loss.backward()
        optimizer.step()

        noise_prediction_loss += loss.item()

        # calculate accuracy
        pred = torch.argmax(pred, dim=1)
        acc += (pred == y).sum().item()
        batch_size = x.size(0)
        acc /= batch_size
    rec_acc.append(acc)
    rec_loss.append(noise_prediction_loss / batch_idx)
    print(f"Epoch {epoch+1}/{epochs} complete! Denoising Loss: {noise_prediction_loss / batch_idx:.4f}, Accuracy: {acc:.4f}")

print("Finish!!")

# save the model
torch.save(model.state_dict(), '3dd.pt')

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
}

# save the experiment data in json
import json
file_name = f"rec/exp_{datetime.now().strftime('%Y_%m_%d_%H_%M')}.json"
with open(file_name, 'w') as f:
    json.dump(exp_data, f)

# generate some images (makegrid) (label min(10,classes))
model.eval()
with torch.no_grad():
    class_labels = torch.randint(0, classes, (test_batch_size,)).to(DEVICE)
    noise = torch.randn(test_batch_size, *img_size).permute(0, 3, 1, 2).to(DEVICE)
    noise = noise / noise.std(dim=(1, 2, 3), keepdim=True)
    noise = noise.detach()
    print(f"noise shape: {noise.shape}, label shape: {class_labels.shape}")
    out = model(noise, class_labels)
    out = out.cpu()
    print(out.shape)
    nrow = int(math.sqrt(test_batch_size))
    out = make_grid(out, nrow=nrow, normalize=True)
    # save the image
    save_image(out, 'generated_images.png')
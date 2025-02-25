import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import numpy as np

from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torch.optim import Adam

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

timestep_embedding_dim = 256
n_layers = 8
hidden_dim = 256
n_timesteps = 1000
beta_minmax=[1e-4, 2e-2]

train_batch_size = 128
inference_batch_size = 64
lr = 5e-5
epochs = 50

seed = 1234

hidden_dims = [hidden_dim for _ in range(n_layers)]
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

discriminator = ghostnet_050(num_classes=10, in_chans=3 if dataset == 'CIFAR10' else 1)
disc_pth = './cifar10_model.pth' if dataset == 'CIFAR10' else './mnist_model.pth'
discriminator.load_state_dict(torch.load(disc_pth, weights_only=True))
for param in discriminator.parameters():
    param.requires_grad = False
discriminator.to(DEVICE)

model = TriDD(
    img_res=img_size,
    label_dim=10,
    proj=8,
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

for epoch in range(epochs):
    noise_prediction_loss = 0
    for batch_idx, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()

        y = y.to(DEVICE)
        noise = torch.randn_like(x).to(DEVICE)
        noise = noise / noise.std(dim=(1, 2, 3), keepdim=True)
        noise = noise.detach()
        # print(noise.shape, y.shape)
        out = model(noise, y)# generated image

        # discriminator loss
        pred = discriminator(out)
        loss = label_sim_loss(pred, y)
        loss.backward()
        optimizer.step()

        noise_prediction_loss += loss.item()

    print("\tEpoch", epoch + 1, "complete!", "\tDenoising Loss: ", noise_prediction_loss / batch_idx)

print("Finish!!")

# save the model
torch.save(model.state_dict(), '3dd.pt')
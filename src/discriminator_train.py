import torch
import torchvision
from torch.utils.data import DataLoader
from timm.models.ghostnet import ghostnet_050
import numpy as np
import argparse

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define data transformations
def get_transforms(dataset):
    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:  # cifar10
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    return transform

def load_data(dataset, batch_size):
    transform = get_transforms(dataset)
    
    if dataset == 'mnist':
        # MNIST has single channel
        train_dataset = torchvision.datasets.MNIST(root='~/datasets', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='~/datasets', train=False, download=True, transform=transform)
    else:  # cifar10
        train_dataset = torchvision.datasets.CIFAR10(root='~/datasets', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='~/datasets', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    train_loss = running_loss / len(train_loader)
    train_acc = 100.0 * correct / total
    return train_loss, train_acc

def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    test_loss = running_loss / len(test_loader)
    test_acc = 100.0 * correct / total
    return test_loss, test_acc

def main():
    parser = argparse.ArgumentParser(description='Train a CNN on MNIST or CIFAR-10')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'cifar10'],
                        help='Dataset to use (mnist or cifar10)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()
    
    # Load data
    train_loader, test_loader = load_data(args.dataset, args.batch_size)
    
    # Initialize model
    # model = SimpleCNN(num_classes=10).to(device)
    model = ghostnet_050(num_classes=10).to(device) if args.dataset == 'cifar10' else ghostnet_050(num_classes=10, in_chans=1).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print(f"Training on {args.dataset} for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    print("Training completed!")
    
    # Save the model
    torch.save(model.state_dict(), f"{args.dataset}_model.pth")
    print(f"Model saved as {args.dataset}_model.pth")

if __name__ == "__main__":
    main()
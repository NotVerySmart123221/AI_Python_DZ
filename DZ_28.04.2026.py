import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scaler = torch.amp.GradScaler('cuda') 

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(
    root="./data", 
    train=True,
    transform=transform,
    download=True
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=256, 
    shuffle=True, 
    num_workers=4, 
    pin_memory=True,
    persistent_workers=True
)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

images, labels = next(iter(train_loader))
images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

show_count = 10

fig, axes = plt.subplots(1, show_count, figsize=(15, 3))

for i in range(show_count):
    img = images[i].detach().cpu() / 2 + 0.5
    npimg = img.numpy()
    axes[i].imshow(np.transpose(npimg, (1, 2, 0)))
    axes[i].axis('off')
    axes[i].set_title(classes[labels[i].item()])

plt.show()

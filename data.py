# Jake Eckfeldt
# 11688261 CPTS 528

# File for loading the data sets, we are using CIFAR 10 a standard benchmark image classifying data set

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# return the loaders for cifar 10
def get_loaders(batch_size=128, num_workers=2, data_dir="./data"):

    # normalizing transformations
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),   
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # load dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )
    
    # wrap data in loader for handling
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_loaders()
    print(" data loaders created successfully!")
    print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")
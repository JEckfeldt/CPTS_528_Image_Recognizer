# Jake Eckfeldt
# 11688261 CPTS 528

# adv_train.py — Adversarial training using PGD attack

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from data import get_loaders
from models import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pgd_attack(model, images, labels, eps=8/255, alpha=2/255, steps=7):
    """Generate PGD adversarial examples for training"""
    ori = images.clone().detach()
    images = images + torch.empty_like(images).uniform_(-eps, eps)
    images = torch.clamp(images, 0, 1)

    for _ in range(steps):
        images.requires_grad = True
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        grad = images.grad.sign()
        images = images + alpha * grad
        delta = torch.clamp(images - ori, -eps, eps)
        images = torch.clamp(ori + delta, 0, 1).detach()
    return images


def train(model, loader, optimizer, device, eps, alpha, steps):
    """one epoch of adversarial training"""
    model.train()
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = pgd_attack(model, images, labels, eps, alpha, steps)

        optimizer.zero_grad()
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def test(model, loader, device):
    """gets accuracy"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


if __name__ == "__main__":
    train_loader, test_loader = get_loaders(batch_size=128)

    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    eps, alpha, steps = 8/255, 2/255, 7  # PGD params

    for epoch in range(num_epochs):
        loss = train(model, train_loader, optimizer, device, eps, alpha, steps)
        acc = test(model, test_loader, device)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {loss:.4f} | Test Accuracy: {acc:.2f}%")

    torch.save(model.state_dict(), "model_cifar10.pth")
    print("adversarial training complete, model saved to model_cifar10.pth")

# Jake Eckfeldt
# 11688261 CPTS 528

# test.py
# File for testing the trained model

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import CNN
from data import get_loaders


def main():
    # setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    _, test_loader = get_loaders(batch_size=100)

    # load saved model
    model = CNN().to(device)
    model.load_state_dict(torch.load("model_cifar10.pth", map_location=device))
    model.eval()
    print("model loaded successfully!")

    # get accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()

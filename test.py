# Jake Eckfeldt
# 11688261 CPTS 528

# test.py — Test a trained CIFAR-10 model with a custom filename

import sys
import torch
from models import CNN
from data import get_loaders


def main():
    # device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get model filename from command line
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "model_cifar10.pth"   # default
    print(f"Loading model from: {model_path}")

    # load data
    _, test_loader = get_loaders(batch_size=100)

    # load model
    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("Model loaded successfully!")

    # calculate accuracy
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

# Jake Eckfeldt
# 11688261 CPTS 528

# train.py — File for running model training

import torch
import torch.nn as nn
import torch.optim as optim
from data import get_loaders
from models import CNN  # Make sure this matches your model filename/class name

# setup torch, use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# training function
def train(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(loader)


# testing function
def test(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


if __name__ == "__main__":
    # load data
    train_loader, test_loader = get_loaders(batch_size=128)

    # initialize model, loss, optimizer
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        loss = train(model, train_loader, optimizer, criterion, device)
        acc = test(model, test_loader, device)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {loss:.4f} | Test Accuracy: {acc:.2f}%")

    # save model
    torch.save(model.state_dict(), "model_cifar10.pth")
    print("training finished, model saved to model_cifar10.pth")

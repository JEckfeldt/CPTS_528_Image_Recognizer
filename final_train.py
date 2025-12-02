# Jake Eckfeldt
# 11688261 CPTS 528

# final_train.py — Adversarial training using PGD attack
# Final version, this uses a PGD attack but also employs techniques to mitigate the accuracy loss of the attack

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

from data import get_loaders
from models import CNN
from adv_train import pgd_attack  # reuse your attack

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# label smoothing
def smooth_labels(labels, num_classes=10, smoothing=0.1):
    confidence = 1.0 - smoothing
    smoothing_value = smoothing / (num_classes - 1)
    one_hot = torch.full((labels.size(0), num_classes), smoothing_value).to(device)
    one_hot.scatter_(1, labels.unsqueeze(1), confidence)
    return one_hot


# step thru trainig
def hybrid_train_step(model, images, labels, optimizer, scaler,
                      eps, alpha, steps, adv_ratio=0.5, label_smooth=0.1):

    model.train()

    batch_size = images.size(0)
    adv_count = int(batch_size * adv_ratio)

    clean_images = images[:batch_size - adv_count]
    clean_labels = labels[:batch_size - adv_count]

    adv_images = images[batch_size - adv_count:]
    adv_labels = labels[batch_size - adv_count:]

    # generate PGD only on part of batch
    adv_images = pgd_attack(model, adv_images, adv_labels, eps, alpha, steps)

    # merge clean + adv
    images_mix = torch.cat([clean_images, adv_images], dim=0)
    labels_mix = torch.cat([clean_labels, adv_labels], dim=0)

    # label smoothing
    labels_smooth = smooth_labels(labels_mix, smoothing=label_smooth)

    optimizer.zero_grad()

    # mixed precision training
    with autocast():
        outputs = model(images_mix)
        loss = torch.mean(torch.sum(-labels_smooth * F.log_softmax(outputs, dim=1), dim=1))

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return loss.item()


# test
def test(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    return 100 * correct / total


# main
if __name__ == "__main__":
    train_loader, test_loader = get_loaders(batch_size=128)

    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    scaler = GradScaler()

    num_epochs = 10

    base_eps = 8/255
    base_alpha = 2/255
    steps = 7

    warmup_epochs = 2          # train clean-only for stability
    finetune_epochs = 2        # clean finetune at end
    label_smoothing = 0.1

    for epoch in range(num_epochs):
        losses = []

        # gradually increase attack strength
        attack_scale = epoch / (num_epochs - finetune_epochs)
        eps = attack_scale * base_eps
        alpha = attack_scale * base_alpha

        # during warmup: clean-only training
        if epoch < warmup_epochs:
            eps = 0
            alpha = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # last epochs: clean-only fine-tuning
            if epoch >= num_epochs - finetune_epochs:
                eps = 0
                alpha = 0
                adv_ratio = 0.0
            else:
                adv_ratio = 0.5

            loss = hybrid_train_step(
                model, images, labels, optimizer, scaler,
                eps, alpha, steps,
                adv_ratio=adv_ratio,
                label_smooth=label_smoothing
            )
            losses.append(loss)

        scheduler.step()
        acc = test(model, test_loader)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {sum(losses)/len(losses):.4f} | Test Acc: {acc:.2f}%")

    torch.save(model.state_dict(), "final_model_cifar10.pth")
    print("modified anti adversarial training finished.")

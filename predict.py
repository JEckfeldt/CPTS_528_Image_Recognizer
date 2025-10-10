# Jake Eckfeldt
# 11688261 CPTS 528

# predict.py — File for guessing individual images on the trained model

# Usage:
#   python predict.py path/to/image.jpg
#   python predict.py path/to/image.jpg --model-path model_cifar10.pth --show

import argparse
import os
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import transforms

# cifar class names
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def load_model(model_path, device):
    from models import CNN as ModelClass

    model = ModelClass().to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def preprocess_image(img_path):
    # resize img 32x32 is cifar compatible
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    tensor = transform(img).unsqueeze(0) 
    return img, tensor


def predict_one(model, img_tensor, device):
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        logits = model(img_tensor)               
        probs = F.softmax(logits, dim=1).cpu() 
        top_prob, top_idx = probs.topk(1, dim=1)
        pred_idx = int(top_idx[0, 0].item())
        confidence = float(top_prob[0, 0].item())
    return pred_idx, confidence


def main():
    parser = argparse.ArgumentParser(description="Predict single image with trained CIFAR-10 model")
    parser.add_argument("image", type=str, help="path to the input image")
    parser.add_argument("--model-path", type=str, default="model_cifar10.pth", help="path to saved model weights")
    parser.add_argument("--show", action="store_true", help="display the image with predicted label")
    parser.add_argument("--device", type=str, default=None, help="device to use, e.g. 'cpu' or 'cuda'. Default autodetect.")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    # device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load model
    model = load_model(args.model_path, device)
    print("Model loaded.")

    # process img
    pil_img, tensor = preprocess_image(args.image)

    # predict
    pred_idx, confidence = predict_one(model, tensor, device)
    label = CIFAR10_CLASSES[pred_idx]
    print(f"Predicted: {label} (class {pred_idx}), confidence: {confidence * 100:.2f}%")

    #  show image with label
    if args.show:
        plt.imshow(pil_img)
        plt.axis("off")
        plt.title(f"{label} ({confidence*100:.1f}%)")
        plt.show()


if __name__ == "__main__":
    main()

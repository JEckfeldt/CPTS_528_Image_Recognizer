# Jake Eckfeldt
# 11688261 CPTS 528

# app.py
# web page for predicting what the image is

import os
from flask import Flask, request, render_template_string
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# --- Load model ---
from models import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model.load_state_dict(torch.load("model_cifar10.pth", map_location=device))
model.eval()

# cifar class names
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# preprocess image
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# simple flask app
app = Flask(__name__)

HTML = """
<!doctype html>
<title>CIFAR-10 Image Classifier</title>
<h1>Upload an image (32x32 preferred)</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
{% if label %}
  <h2>Prediction: {{ label }} ({{ conf }}% confidence)</h2>
  <img src="{{ url }}" style="max-width:200px;">
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def upload_predict():
    label = None
    conf = None
    url = None

    if request.method == "POST":
        file = request.files.get("file")
        if file:
            img_path = os.path.join("static", file.filename)
            os.makedirs("static", exist_ok=True)
            file.save(img_path)

            # Preprocess image
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)

            # Predict
            with torch.no_grad():
                outputs = model(tensor)
                probs = F.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)
                label = CLASSES[pred.item()]
                conf = round(conf.item() * 100, 2)
                url = "/" + img_path

    return render_template_string(HTML, label=label, conf=conf, url=url)


if __name__ == "__main__":
    app.run(debug=True)

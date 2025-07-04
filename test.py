import torch
import torch.nn as nn
from PIL import Image
import os
from torchvision import transforms

# === Define the CViT model (must match your training)
class CViT(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.embedding = nn.Linear(128 * 16 * 16, 512)
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(128, num_classes), nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.embedding(x).unsqueeze(0)  # [1, batch, dim]
        x = self.transformer(x).squeeze(0)
        return self.classifier(x)

# === Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CViT(num_classes=10).to(device)
model.load_state_dict(torch.load("best_model_cvit.pt", map_location=device))
model.eval()

# === Define transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # RGB normalization
])

# === Predict on all images in the folder
img_folder = "mnist_digits"
for i in range(10):
    img_path = os.path.join(img_folder, f"digit_{i}.png")
    img = Image.open(img_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred = output.argmax(1).item()

    print(f"🖼 digit_{i}.png → Predicted: {pred}")

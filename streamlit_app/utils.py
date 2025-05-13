# streamlit_app/utils.py

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# CNN model architecture (same as training)
class PersonalityCNN(nn.Module):
    def __init__(self):
        super(PersonalityCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 5)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

# Map prediction index to label
trait_labels = ["Agreeableness", "Conscientiousness", "Extraversion", "Neuroticism", "Openness"]

# Prediction function
def predict_personality(image_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PersonalityCNN().to(device)
    model.load_state_dict(torch.load("/Users/sakshizanjad/Desktop/project_root/models/personality_cnn.pth", map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = Image.open(image_file).convert("L")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return trait_labels[predicted.item()]
# scripts/model_training.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

# ✅ Data augmentation + preprocessing transforms
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.RandomApply([
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(degrees=0, shear=5),
        transforms.RandomResizedCrop(128, scale=(0.9, 1.0)),
    ], p=0.6),  # Apply augmentation to ~60% of images

    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ✅ Load dataset
data_dir = '/Users/sakshizanjad/Desktop/ScriptSense--1/Dataset/processed_samples'
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ✅ Calculate class weights
labels = [label for _, label in dataset.samples]
classes = np.unique(labels)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

print("✅ Class distribution:", dict(zip(classes, np.bincount(labels))))
print("⚖️  Class weights:", dict(zip(classes, class_weights)))

# ✅ CNN Model
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
            nn.Linear(256, 5)  # 5 personality traits
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

# ✅ Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PersonalityCNN().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ Training loop
epochs = 200
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# ✅ Save the model
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), '/Users/sakshizanjad/Desktop/ScriptSense--1/scripts/models/personality_cnn.pth')
print("✅ Model training complete and saved.")


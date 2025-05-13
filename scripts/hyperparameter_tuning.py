import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from skorch import NeuralNetClassifier
import numpy as np

# Define transformations
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load dataset
data_dir = '/Users/sakshizanjad/Desktop/ScriptSense--1/Dataset/processed_samples'
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
X = [data[0].numpy() for data in dataset]
y = [data[1] for data in dataset]
X = np.stack(X)
X = X.reshape(-1, 1, 128, 128)  # Reshape for skorch
y = np.array(y)

# Compute class weights
classes = np.unique(y)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

print("✅ Class distribution:", dict(zip(classes, np.bincount(y))))
print("⚖️  Class weights:", dict(zip(classes, class_weights)))

# CNN Model with hyperparams
class PersonalityCNN(nn.Module):
    def __init__(self, conv1_out=16, conv2_out=32, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, conv1_out, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(conv2_out, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Define custom loss function using class weights
def weighted_loss(y_pred, y_true):
    return nn.CrossEntropyLoss(weight=class_weights_tensor.to(y_pred.device))(y_pred, y_true)

# Wrap with skorch
net = NeuralNetClassifier(
    PersonalityCNN,
    max_epochs=10,
    lr=0.001,
    optimizer=torch.optim.Adam,
    criterion=weighted_loss,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Define grid
params = {
    'lr': [0.01, 0.001],
    'optimizer': [torch.optim.Adam, torch.optim.SGD],
    'batch_size': [16, 32],
    'module__conv1_out': [16, 32],
    'module__conv2_out': [32, 64],
    'module__dropout': [0.3, 0.5],
}

# Run grid search
gs = GridSearchCV(net, params, refit=True, cv=3, scoring='accuracy', verbose=2)
gs.fit(X, y)

# Best model and score
print("✅ Best Score:", gs.best_score_)
print("✅ Best Params:", gs.best_params_)

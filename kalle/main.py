import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# -----------------------------
# 1. Load Data (example: CIFAR-10)
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.ImageFolder(root='dataset/asl_alphabet_train', transform=transform)
val_dataset = datasets.ImageFolder(root='dataset/asl_alphabet_test', transform=transform)

trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
testloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# -----------------------------
# 2. Define CNN Model
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # 3xHxW -> 16xHxW
            nn.ReLU(),
            nn.MaxPool2d(2, 2),              # 16xH/2xW/2

            nn.Conv2d(16, 32, 3, padding=1), # 32xH/2xW/2 -> 32xH/4xW/4
            nn.ReLU(),
            nn.MaxPool2d(2, 2)               # 32xH/4xW/4
        )

        # Dynamically compute the flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 128, 128)  # Change 128x128 to your actual input size
            dummy_output = self.conv(dummy_input)
            self.flattened_size = dummy_output.view(-1).shape[0]

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, 27)  # 27 classes for your ASL dataset
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# -----------------------------
# 3. Train the Model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):  # Train for 5 epochs
    running_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {running_loss:.3f}")

# -----------------------------
# 4. Evaluate the Model
# -----------------------------
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test set: {100 * correct / total:.2f}%")

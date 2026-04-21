import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ----- Parameters -----
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
DATASET_PATH = "dataset"

# ----- Preprocessing -----
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ----- Dataset -----
dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

class_names = dataset.classes
print("Classes:", class_names)

# ----- Model -----
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(class_names))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ----- Loss + Optimizer -----
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ----- Training Loop -----
for epoch in range(EPOCHS):

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Loss: {running_loss:.3f} | Accuracy: {accuracy:.2f}%")

# ----- Save Model -----
torch.save(model.state_dict(), "coral_model.pth")

print("Model saved as coral_model.pth")

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Image size expected by many CNNs
IMG_SIZE = 227
BATCH_SIZE = 32

# --- Image Transformations / Preprocessing ---
transform = transforms.Compose([
    
    # Resize images to same dimensions
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    
    # Data augmentation to improve model robustness
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    
    # Convert image to tensor
    transforms.ToTensor(),
    
    # Normalize RGB channels
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --- Load Dataset ---
dataset = datasets.ImageFolder(
    root="dataset",
    transform=transform
)

# --- Data Loader ---
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)

# --- Class Labels ---
print(dataset.classes)
# ['dead', 'dying', 'healthy']

# --- Example Batch ---
for images, labels in dataloader:
    print(images.shape)   # e.g. [32, 3, 224, 224]
    print(labels.shape)
    break

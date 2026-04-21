import torch
from torchvision import transforms, models
from PIL import Image

# ----- Parameters -----
IMG_SIZE = 224
MODEL_PATH = "coral_model.pth"

classes = ['dead', 'dying', 'healthy']

# ----- Image preprocessing -----
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ----- Load Model -----
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 3)

model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# ----- Load Image -----
image_path = "test_coral.jpg"

image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0)

# ----- Prediction -----
with torch.no_grad():

    outputs = model(image)

    _, predicted = torch.max(outputs, 1)

    prediction = classes[predicted.item()]

print("Prediction:", prediction)

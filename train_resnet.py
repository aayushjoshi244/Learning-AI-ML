import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import pandas as pd
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset path and config
data_dir = "office31/webcam"
num_classes = 31
batch_size = 32
num_epochs = 10

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load dataset
train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Load pretrained ResNet18 and modify the final layer
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Lists to store training metrics
epoch_list = []
loss_list = []
accuracy_list = []

print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct / total

    # Store for Excel log
    epoch_list.append(epoch + 1)
    loss_list.append(epoch_loss)
    accuracy_list.append(epoch_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

# Save model
torch.save(model.state_dict(), "Office31_resnet.pth")
print("✅ Model Saved as Office31_resnet.pth")

# Save training log to Excel
df_log = pd.DataFrame({
    "Epoch": epoch_list,
    "Loss": loss_list,
    "Accuracy": accuracy_list
})
df_log.to_excel("office31_training_log.xlsx", index=False)
print("📊 Training log saved as office31_training_log.xlsx")

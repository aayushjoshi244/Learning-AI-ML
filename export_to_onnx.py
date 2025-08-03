import torch
from torchvision import models
import torch.nn as nn

num_classes = 31  # ✅ Match the number of classes in the model
model_path = "Office31_resnet.pth"
onnx_path = "Office31_resnet.onnx"

# Load model with correct number of output classes
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load trained weights
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# Dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(model, dummy_input, onnx_path,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'},
                                'output': {0: 'batch_size'}})

print(f"✅ Model exported to {onnx_path}")

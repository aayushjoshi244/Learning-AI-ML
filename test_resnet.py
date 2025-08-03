import onnxruntime as ort
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms

onnx_model_path = "Office31_resnet_simplified.onnx"
test_dir = "dataset_subset/test"  # contains only 6 folders

# All class names in training order
all_class_names = [
    'back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator',
    'desk_chair', 'desk_lamp', 'desktop_computer', 'file_cabinet', 'flip_flops',
    'headphones', 'keyboard', 'laptop_computer', 'letter_tray', 'mobile_phone',
    'monitor', 'mouse', 'mug', 'paper_notebook', 'pen', 'phone', 'printer',
    'projector', 'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker',
    'stapler', 'tape_dispenser'
]

target_classes = ['back_pack', 'bike', 'pen', 'bookcase', 'desk_chair', 'monitor']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

session = ort.InferenceSession(onnx_model_path)
input_name = session.get_inputs()[0].name

correct = 0
total = 0

for cls in target_classes:
    folder = os.path.join(test_dir, cls)
    if not os.path.isdir(folder):
        print(f"⚠️ Folder not found: {folder}")
        continue
    for img_name in os.listdir(folder):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img_path = os.path.join(folder, img_name)
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).numpy()

        output = session.run(None, {input_name: input_tensor})
        pred = np.argmax(output[0])
        pred_label = all_class_names[pred]

        is_correct = (pred_label == cls)
        total += 1
        correct += int(is_correct)

        print(f"[{cls}] ➝ Predicted: {pred_label} {'✅' if is_correct else '❌'}")

accuracy = correct / total if total > 0 else 0
print(f"\n🎯 Accuracy on 6 target classes: {accuracy * 100:.2f}%")

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
import time

# -------------------------------
# Paths & Settings
# -------------------------------
engine_path = "office31.engine"
test_dir = "dataset_subset/test"
target_classes = ['back_pack', 'bike', 'pen', 'bookcase', 'desk_chair', 'monitor']
all_class_names = [
    'back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator',
    'desk_chair', 'desk_lamp', 'desktop_computer', 'file_cabinet', 'flip_flops',
    'headphones', 'keyboard', 'laptop_computer', 'letter_tray', 'mobile_phone',
    'monitor', 'mouse', 'mug', 'paper_notebook', 'pen', 'phone', 'printer',
    'projector', 'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker',
    'stapler', 'tape_dispenser'
]
class_to_index = {cls: idx for idx, cls in enumerate(all_class_names)}

# -------------------------------
# Preprocessing
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------------
# TensorRT Utilities
# -------------------------------
TRT_LOGGER = trt.Logger()

def load_engine(engine_file_path):
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append((host_mem, device_mem))
        else:
            outputs.append((host_mem, device_mem))
    return inputs, outputs, bindings, stream

def infer(context, bindings, inputs, outputs, stream):
    [cuda.memcpy_htod_async(inp[1], inp[0], stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out[0], out[1], stream) for out in outputs]
    stream.synchronize()
    return outputs[0][0:31]

# -------------------------------
# Run Inference
# -------------------------------
engine = load_engine(engine_path)
context = engine.create_execution_context()
inputs, outputs, bindings, stream = allocate_buffers(engine)

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

        image_path = os.path.join(folder, img_name)
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).numpy().astype(np.float32)

        np.copyto(inputs[0][0], input_tensor.ravel())

        output = infer(context, bindings, inputs, outputs, stream)
        pred = np.argmax(output)
        pred_label = all_class_names[pred]

        total += 1
        correct += int(pred_label == cls)

        print(f"[{cls}] ➝ Predicted: {pred_label} {'✅' if pred_label == cls else '❌'}")

accuracy = correct / total if total > 0 else 0
print(f"\n🎯 TensorRT Accuracy on 6 classes: {accuracy * 100:.2f}%")

from onnxsim import simplify
import onnx

model_path = "Office31_resnet.onnx"
optimized_path = "Office31_resnet_simplified.onnx"

model = onnx.load(model_path)
model_simplified, check = simplify(model)

onnx.save(model_simplified, optimized_path)

if check:
    print(f"✅ Simplified model saved as {optimized_path}")
else:
    print("⚠️ Simplification check failed. The model may not be simplified correctly.")
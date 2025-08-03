import onnx
import onnxoptimizer

model_path = "Office31_resnet.onnx"
model = onnx.load(model_path)

passes = onnxoptimizer.get_available_passes()
optimized_model = onnxoptimizer.optimize(model, passes)

optimized_path = "Office31_resnet_optimized.onnx"
onnx.save(optimized_model, optimized_path)

print(f"✅ Optimized model saved as {optimized_path}")
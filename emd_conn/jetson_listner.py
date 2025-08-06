import paho.mqtt.client as mqtt
import onnxruntime as ort
import numpy as np
import cv2
from pathlib import Path

# Load ONNX model
session = ort.InferenceSession("office31_model.onnx")  # update with your model filename
input_name = session.get_inputs()[0].name

# Your label mapping
class_names = ['back_pack', 'bike', 'bookcase', 'desk_chair', 'monitor', 'pen']

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # CHW format
    img = np.expand_dims(img, axis=0)  # batch dimension
    return img

def classify_image(img_path):
    input_tensor = preprocess_image(img_path)
    outputs = session.run(None, {input_name: input_tensor})
    pred = np.argmax(outputs[0])
    return class_names[pred]

def on_message(client, userdata, msg):
    print(f"[MQTT] Received message: {msg.payload.decode()} on topic: {msg.topic}")
    
    if msg.payload.decode() == "image_ready":
        image_path = "capture.jpg"
        if Path(image_path).exists():
            label = classify_image(image_path)
            print(f"[INFO] Detected object: {label}")
            client.publish("iot/results", label)
        else:
            print("[ERROR] capture.jpg not found.")

def main():
    client = mqtt.Client()
    client.on_message = on_message

    client.connect("broker.hivemq.com", 1883, 60)
    client.subscribe("iot/trigger")
    print("[MQTT] Listening for 'image_ready' on topic iot/trigger")

    client.loop_forever()

if __name__ == "__main__":
    main()

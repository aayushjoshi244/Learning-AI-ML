import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if ret:
    cv2.imwrite("capture.jpg", frame)
    print("Image captured")

# Now send MQTT or HTTP trigger
import paho.mqtt.publish as publish
publish.single("iot/trigger", "image_ready", hostname="broker.hivemq.com")

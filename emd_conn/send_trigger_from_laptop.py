import cv2
import paho.mqtt.publish as publish
import os

# CONFIG: Replace these with your Jetson details
JETSON_USER = "aayushjoshi244"                     
JETSON_IP = "192.168.1.38"                
DEST_PATH = f"/home/{JETSON_USER}/Desktop/kmitl"        

cap = cv2.VideoCapture(1)  # Change to 0 if 1 doesn’t work

print("[LAPTOP] Press 'q' to capture image...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Couldn't read from webcam.")
        break

    cv2.imshow("Live Feed - Press 'q' to Capture", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        filename = "capture.jpg"
        cv2.imwrite(filename, frame)
        print("[LAPTOP] Image captured.")

        # Send image to Jetson using scp
        scp_command = f"scp {filename} {JETSON_USER}@{JETSON_IP}:{DEST_PATH}"
        result = os.system(scp_command)
        if result == 0:
            print("[LAPTOP] Image sent to Jetson.")
        else:
            print("[ERROR] Failed to send image to Jetson.")
            break

        # Send MQTT trigger
        publish.single("iot/aayush244/blast", "image_ready", hostname="broker.hivemq.com")
        print("[LAPTOP] MQTT trigger sent: image_ready")
        break

cap.release()
cv2.destroyAllWindows()

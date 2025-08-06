import paho.mqtt.client as mqtt

def on_message(client, userdata, msg):
    print(f"[LAPTOP] Inference result: {msg.payload.decode()}")

client = mqtt.Client()
client.on_message = on_message

client.connect("broker.hivemq.com", 1883, 60)
client.subscribe("iot/results")
print("[LAPTOP] Listening for results on topic iot/results...")

client.loop_forever()

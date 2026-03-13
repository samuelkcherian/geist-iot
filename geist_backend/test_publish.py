import paho.mqtt.client as mqtt
import json
import time

broker = "192.168.1.4"
topic = "home/espectre/node1"
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
client.connect(broker, 1883, 60)

# Simulate motion (Fall)
print("Simulating Motion/Fall...")
client.publish(topic, json.dumps({"state": "motion"}))
time.sleep(5)

# Simulate idle (Safe)
print("Simulating Idle/Safe...")
client.publish(topic, json.dumps({"state": "idle"}))
client.disconnect()

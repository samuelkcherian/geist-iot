import paho.mqtt.client as mqtt
import json
import time

broker = "192.168.1.4"
topic = "home/espectre/node1"
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
client.connect(broker, 1883, 60)

# Simulate Empty (Safe)
print("Simulating Empty (Safe)...")
client.publish(topic, json.dumps({"state": "empty"}))
time.sleep(3)

# Simulate Walk
print("Simulating Walk...")
client.publish(topic, json.dumps({"state": "walk"}))
time.sleep(3)

# Simulate Sit
print("Simulating Sit...")
client.publish(topic, json.dumps({"state": "sit"}))
time.sleep(3)

# Simulate Fall
print("Simulating Fall...")
client.publish(topic, json.dumps({"state": "fall"}))
time.sleep(5)

# Reset to Empty
print("Resetting to Empty...")
client.publish(topic, json.dumps({"state": "empty"}))

client.disconnect()

import sqlite3
from flask import Flask, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from datetime import datetime
import json
import threading
import paho.mqtt.client as mqtt

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'geist_secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', logger=True, engineio_logger=True)

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect('geist.db')
    c = conn.cursor()
    # Create table if not exists
    c.execute('''CREATE TABLE IF NOT EXISTS logs 
                 (id INTEGER PRIMARY KEY, event_type TEXT, timestamp TEXT)''')
    conn.commit()
    conn.close()

# Initialize DB immediately
init_db()

# --- HELPER: SAVE TO DB ---
def log_event(event_type):
    conn = sqlite3.connect('geist.db')
    c = conn.cursor()
    # Get current time (e.g., "10:30 AM")
    time_str = datetime.now().strftime("%b %d, %I:%M %p")
    c.execute("INSERT INTO logs (event_type, timestamp) VALUES (?, ?)", (event_type, time_str))
    conn.commit()
    conn.close()

# --- MQTT SETUP ---
MQTT_BROKER = "192.168.1.4"
MQTT_PORT = 1883
MQTT_TOPIC = "home/espectre/node1"

def on_mqtt_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT broker with result code {rc}")
    client.subscribe(MQTT_TOPIC)

def on_mqtt_message(client, userdata, msg):
    try:
        payload = msg.payload.decode('utf-8')
        data = json.loads(payload)
        state = data.get("state")
        
        if state in ["fall", "motion"]:
            print("MQTT: Fall/Motion detected! Triggering FALL state.")
            socketio.emit('status_update', {'status': 'FALL', 'color': 'red'})
            log_event("Fall Detected")
        elif state == "walk":
            print("MQTT: Walk detected.")
            socketio.emit('status_update', {'status': 'WALK', 'color': 'yellow'})
        elif state == "sit":
            print("MQTT: Sit detected.")
            socketio.emit('status_update', {'status': 'SIT', 'color': 'blue'})
        elif state in ["empty", "idle"]:
            print("MQTT: Room Empty/Idle. Safe state.")
            socketio.emit('status_update', {'status': 'EMPTY', 'color': 'green'})
    except Exception as e:
        print(f"Error parsing MQTT message: {e}")

mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
mqtt_client.on_connect = on_mqtt_connect
mqtt_client.on_message = on_mqtt_message

def start_mqtt():
    try:
        print(f"Connecting to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}...")
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_forever()
    except Exception as e:
        print(f"MQTT Connection failed: {e}")

# Start MQTT client in a background thread
mqtt_thread = threading.Thread(target=start_mqtt, daemon=True)
mqtt_thread.start()

@app.route('/')
def index():
    return "Geist Database Backend Running!"

# --- API: FETCH HISTORY FOR APP ---
@app.route('/api/logs')
def get_logs():
    conn = sqlite3.connect('geist.db')
    conn.row_factory = sqlite3.Row # Allows accessing columns by name
    c = conn.cursor()
    # Get last 20 events, newest first
    c.execute("SELECT * FROM logs ORDER BY id DESC LIMIT 20")
    rows = c.fetchall()
    conn.close()
    
    # Convert to JSON format for Flutter
    results = [{"event": r["event_type"], "time": r["timestamp"]} for r in rows]
    return jsonify(results)

# --- TRIGGER EVENTS (Simulate) ---
@app.route('/trigger/<event_type>')
def trigger_event(event_type):
    if event_type == "fall":
        # 1. Alert the Phone (Real-time)
        socketio.emit('status_update', {'status': 'FALL', 'color': 'red'})
        # 2. Save to Database (History)
        log_event("Fall Detected")
        return "FALL Triggered & Saved!"
    
    elif event_type == "safe":
        socketio.emit('status_update', {'status': 'Safe', 'color': 'green'})
        # We usually don't log 'safe' states, only alerts, but you can if you want
        return "System Reset to Safe"
        
    return "Unknown event"

if __name__ == '__main__':
    # host='0.0.0.0' allows external connection
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
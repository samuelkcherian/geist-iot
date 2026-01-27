import sqlite3
from flask import Flask, jsonify
from flask_socketio import SocketIO, emit
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'geist_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

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
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
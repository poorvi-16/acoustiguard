# tdoa/server.py
# The central brain — receives detection events
# from all nodes, runs TDOA, feeds the dashboard

from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
import time
import threading
from collections import defaultdict
from trilateration import calculate_drone_position

app    = Flask(__name__,
               template_folder='../dashboard/templates',
               static_folder='../dashboard/static')
socketio = SocketIO(app, cors_allowed_origins="*")

# Store recent events grouped by time window
# key = time window, value = list of events
pending_events = defaultdict(list)
drone_history  = []  # all confirmed positions
lock           = threading.Lock()

WINDOW_SECONDS = 2.0  # group events within 2 seconds


def process_pending_events():
    """
    Runs every second in background.
    Groups events that arrived close together
    and runs TDOA on groups of 3+.
    """
    while True:
        time.sleep(1.0)
        now = time.time()

        with lock:
            # Find event groups old enough to process
            windows_to_process = [
                w for w in pending_events
                if now - w > WINDOW_SECONDS
            ]

            for window in windows_to_process:
                events = pending_events.pop(window)

                if len(events) >= 3:
                    print(f"\nProcessing {len(events)} "
                          f"events from window {window:.1f}")

                    position = calculate_drone_position(events)

                    if position:
                        position['timestamp'] = now
                        drone_history.append(position)

                        # Keep only last 200 positions
                        if len(drone_history) > 200:
                            drone_history.pop(0)

                        # Push to dashboard in real time
                        socketio.emit(
                            'drone_position', position
                        )
                        print(f"Emitted position to dashboard: "
                              f"({position['lat']:.5f}, "
                              f"{position['lon']:.5f})")
                else:
                    print(f"Not enough nodes in window "
                          f"({len(events)}/3 minimum)")


@app.route('/event', methods=['POST'])
def receive_event():
    """
    Receives a detection event from one node.
    Groups it with other events in the same time window.
    """
    data = request.json

    if not data:
        return jsonify({"error": "No data"}), 400

    # Handle both single event and batch
    events = data.get('events', [data])

    with lock:
        for event in events:
            # Round timestamp to nearest 2-second window
            ts     = event.get('timestamp', time.time())
            window = round(ts / WINDOW_SECONDS) * WINDOW_SECONDS
            pending_events[window].append(event)

    return jsonify({
        "status":   "received",
        "events":   len(events),
        "queue":    sum(len(v)
                        for v in pending_events.values())
    })


@app.route('/positions')
def get_positions():
    """Returns last 50 drone positions for the map."""
    return jsonify(drone_history[-50:])


@app.route('/status')
def get_status():
    """Health check endpoint."""
    return jsonify({
        "status":         "running",
        "positions_logged": len(drone_history),
        "pending_events": sum(len(v)
                              for v in pending_events.values())
    })


@app.route('/')
def dashboard():
    """Serves the live map dashboard."""
    return render_template('index.html')


# Start background event processor
processor = threading.Thread(
    target=process_pending_events,
    daemon=True
)
processor.start()

if __name__ == '__main__':
    print("=" * 50)
    print("AcoustiGuard Server Starting")
    print("Dashboard: http://localhost:5000")
    print("=" * 50)
    socketio.run(app, debug=False, port=5000)
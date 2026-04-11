# demo/live_demo.py
# Listens to your real microphone
# When it hears a drone sound it triggers
# the simulation on the dashboard
# Play a drone YouTube video near your laptop to test

import sys
import os
import time
import threading
import requests
import numpy as np

# Add parent folder to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))

from detection.audio_capture import start_listening, get_audio_chunk
from detection.classifier import DroneClassifier
from demo.simulate import send_detection, WAYPOINTS

SERVER   = "http://localhost:5000/event"
COOLDOWN = 3.0  # seconds between detections

print("=" * 50)
print("AcoustiGuard — Live Microphone Demo")
print("=" * 50)
print("Loading classifier...")

clf = DroneClassifier()

print("Microphone listening...")
print("Play a drone sound near your laptop")
print("Press Ctrl+C to stop")
print("-" * 50)

stream         = start_listening()
last_detection = 0
waypoint_index = 0

try:
    while True:
        # Get 1 second of audio from mic
        chunk = get_audio_chunk()

        # Classify it
        is_drone, confidence = clf.predict(chunk)

        # Show live volume bar
        volume = int(np.abs(chunk).mean() * 500)
        bar    = "█" * min(volume, 30)
        status = "🚨 DRONE" if is_drone else "  clear"
        print(f"\r{status} | {confidence*100:5.1f}% | "
              f"{bar:<30}", end="", flush=True)

        now = time.time()

        if is_drone and (now - last_detection) > COOLDOWN:
            last_detection = now
            print(f"\n\n🚨 DRONE DETECTED — "
                  f"{confidence*100:.1f}% confidence")
            print(f"Sending position to dashboard...")

            # Use next waypoint in flight path
            lat, lon = WAYPOINTS[
                waypoint_index % len(WAYPOINTS)
            ]
            waypoint_index += 1

            # Send to server in background thread
            threading.Thread(
                target=send_detection,
                args=(lat, lon, "DJI Phantom 4"),
                daemon=True
            ).start()

            print(f"Position sent: ({lat:.5f}, {lon:.5f})")
            print("-" * 50)

except KeyboardInterrupt:
    print("\n\nStopping microphone...")
    stream.stop()
    stream.close()
    print("Done.")

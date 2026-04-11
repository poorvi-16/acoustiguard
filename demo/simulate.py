# demo/simulate.py
# Simulates ONE drone flying across IISc campus
# Sends realistic TDOA events to the server

import time
import math
import random
import requests

SERVER = "http://localhost:5000/event"
SPEED_OF_SOUND = 343.0

# Three IoT nodes — IISc campus
NODES = [
    {"id":"node_a", "lat":12.97160, "lon":77.59460},
    {"id":"node_b", "lat":12.97450, "lon":77.59700},
    {"id":"node_c", "lat":12.96980, "lon":77.59750},
]

# Drone flight path — one drone, one path
WAYPOINTS = [
    (12.9705, 77.5935),
    (12.9710, 77.5940),
    (12.9714, 77.5944),
    (12.9718, 77.5947),
    (12.9722, 77.5951),
    (12.9726, 77.5955),
    (12.9730, 77.5959),
    (12.9728, 77.5963),
    (12.9724, 77.5960),
    (12.9718, 77.5955),
]

# Fixed drone ID — always the same drone
DRONE_ID = "DRONE_001"

def haversine(lat1, lon1, lat2, lon2):
    R    = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a    = (math.sin(dphi/2)**2 +
            math.cos(phi1)*math.cos(phi2)*
            math.sin(dlam/2)**2)
    return 2*R*math.asin(math.sqrt(a))


def send_detection(drone_lat, drone_lon):
    base_time = time.time()
    events    = []

    for node in NODES:
        dist        = haversine(drone_lat, drone_lon,
                                node['lat'], node['lon'])
        travel_time = dist / SPEED_OF_SOUND
        jitter      = random.uniform(-0.003, 0.003)

        events.append({
            "node_id":    node['id'],
            "timestamp":  base_time + travel_time + jitter,
            "node_lat":   node['lat'],
            "node_lon":   node['lon'],
            "drone_model":"DJI Phantom 4",
            "confidence": round(random.uniform(0.86,0.97), 2),
            "drone_id":   DRONE_ID,   # same ID every time
        })

    try:
        r = requests.post(SERVER,
                          json={"events": events},
                          timeout=3)
        if r.status_code == 200:
            print(f"  Sent — server queued: "
                  f"{r.json().get('queue', 0)}")
    except Exception as e:
        print(f"  Cannot reach server: {e}")
        print(f"  Make sure server.py is running first!")


def run_simulation():
    print("=" * 50)
    print("AcoustiGuard — Single Drone Simulation")
    print("=" * 50)
    print(f"Drone ID:    {DRONE_ID}")
    print(f"Model:       DJI Phantom 4")
    print(f"Waypoints:   {len(WAYPOINTS)}")
    print()

    time.sleep(2)

    for i, (lat, lon) in enumerate(WAYPOINTS):
        print(f"Waypoint {i+1}/{len(WAYPOINTS)}: "
              f"({lat:.5f}, {lon:.5f})")
        send_detection(lat, lon)
        time.sleep(3)

    print("\n" + "=" * 50)
    print("Simulation complete!")
    print("Check dashboard at http://localhost:5000")
    print("=" * 50)


if __name__ == "__main__":
    run_simulation()
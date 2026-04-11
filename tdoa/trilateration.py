# tdoa/trilateration.py
# The math that figures out WHERE the drone is
# using time differences between 3 microphone nodes
#
# Simple explanation:
# Sound from drone reaches Node A at time 1.000s
# Sound from drone reaches Node B at time 1.004s  
# Sound from drone reaches Node C at time 1.007s
# From these differences we calculate the drone position

import numpy as np
from scipy.optimize import fsolve
import math

SPEED_OF_SOUND = 343.0  # metres per second at 20°C

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculates distance in metres between
    two GPS coordinates.
    Same formula used in Google Maps.
    """
    R     = 6371000  # Earth radius in metres
    phi1  = math.radians(lat1)
    phi2  = math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)

    a = (math.sin(dphi/2)**2 +
         math.cos(phi1) * math.cos(phi2) *
         math.sin(dlam/2)**2)

    return 2 * R * math.asin(math.sqrt(a))


def tdoa_equations(guess, nodes, time_diffs):
    """
    The core TDOA math.
    
    We guess a drone position, calculate what the
    time differences WOULD be at that position,
    and compare to what we actually measured.
    
    scipy.optimize.fsolve adjusts the guess until
    the calculated differences match the measured ones.
    """
    lat, lon = guess
    equations = []

    # Use first node as reference
    ref_lat, ref_lon = nodes[0]
    ref_dist = haversine(lat, lon, ref_lat, ref_lon)

    for i in range(1, len(nodes)):
        node_lat, node_lon = nodes[i]
        dist_i = haversine(lat, lon, node_lat, node_lon)

        # What TDOA would we expect at this guess position?
        predicted_tdoa = (dist_i - ref_dist) / SPEED_OF_SOUND

        # Difference from what we actually measured
        equations.append(predicted_tdoa - time_diffs[i-1])

    return equations


def calculate_drone_position(events):
    """
    Main function — takes detection events from 3+ nodes
    and returns the drone's GPS position.
    
    events: list of dicts, each containing:
      - node_id:    string identifier
      - timestamp:  float, Unix time in seconds
      - node_lat:   float, node GPS latitude
      - node_lon:   float, node GPS longitude
      - confidence: float, 0-1 detection confidence
    
    Returns dict with lat, lon, accuracy_m
    or None if calculation fails
    """
    if len(events) < 3:
        print(f"Need 3+ nodes, got {len(events)}")
        return None

    # Sort events by timestamp — earliest first
    events_sorted = sorted(events, key=lambda x: x['timestamp'])

    # Extract node positions and timestamps
    nodes      = [(e['node_lat'], e['node_lon'])
                   for e in events_sorted]
    timestamps = [e['timestamp'] for e in events_sorted]

    # Calculate time differences relative to first node
    time_diffs = [timestamps[i] - timestamps[0]
                  for i in range(1, len(timestamps))]

    print(f"\nTDOA Calculation:")
    print(f"  Nodes used: {len(nodes)}")
    for i, td in enumerate(time_diffs):
        print(f"  Node {i+2} time diff: {td*1000:.2f}ms "
              f"= {td*SPEED_OF_SOUND:.1f}m sound travel diff")

    # Initial guess — centroid of all nodes
    init_lat = np.mean([n[0] for n in nodes])
    init_lon = np.mean([n[1] for n in nodes])

    # Solve the equations
    try:
        solution, info, ier, msg = fsolve(
            tdoa_equations,
            [init_lat, init_lon],
            args=(nodes, time_diffs),
            full_output=True
        )

        if ier != 1:
            print(f"TDOA solver warning: {msg}")

        solved_lat = float(solution[0])
        solved_lon = float(solution[1])

        # Calculate average confidence
        avg_confidence = float(np.mean(
            [e['confidence'] for e in events]
        ))

        # Estimate accuracy based on timing jitter
        # ±5ms NTP jitter → ±1.7m per node → ~±18m combined
        accuracy_m = 18.0

        result = {
            "lat":        solved_lat,
            "lon":        solved_lon,
            "accuracy_m": accuracy_m,
            "confidence": avg_confidence,
            "nodes_used": len(nodes),
            "drone_model": events[0].get(
                'drone_model', 'Unknown UAV'
            )
        }

        print(f"  Position: ({solved_lat:.6f}, "
              f"{solved_lon:.6f})")
        print(f"  Accuracy: ±{accuracy_m}m")

        return result

    except Exception as e:
        print(f"TDOA calculation failed: {e}")
        return None


def test_trilateration():
    """
    Tests TDOA with a known drone position.
    We know the drone is at a specific point —
    verify our math gives back that same point.
    """
    print("=" * 50)
    print("Testing TDOA Trilateration")
    print("=" * 50)

    # IISc campus coordinates
    # Node positions — 3 corners of campus
    NODE_A = (12.97160, 77.59460)  # Main gate
    NODE_B = (12.97450, 77.59700)  # Library
    NODE_C = (12.96980, 77.59750)  # Sports complex

    # KNOWN drone position — we will verify
    # our math recovers this
    DRONE_TRUE = (12.97200, 77.59550)

    print(f"\nTrue drone position:  {DRONE_TRUE}")
    print(f"Node A (Main Gate):   {NODE_A}")
    print(f"Node B (Library):     {NODE_B}")
    print(f"Node C (Sports):      {NODE_C}")

    # Calculate TRUE distances
    dist_a = haversine(*DRONE_TRUE, *NODE_A)
    dist_b = haversine(*DRONE_TRUE, *NODE_B)
    dist_c = haversine(*DRONE_TRUE, *NODE_C)

    print(f"\nDistances:")
    print(f"  Drone → Node A: {dist_a:.1f}m")
    print(f"  Drone → Node B: {dist_b:.1f}m")
    print(f"  Drone → Node C: {dist_c:.1f}m")

    # Calculate TRUE arrival times
    base_time = 1000.0  # arbitrary base timestamp
    time_a    = base_time + dist_a / SPEED_OF_SOUND
    time_b    = base_time + dist_b / SPEED_OF_SOUND
    time_c    = base_time + dist_c / SPEED_OF_SOUND

    # Add realistic NTP jitter (±3ms)
    import random
    time_a += random.uniform(-0.003, 0.003)
    time_b += random.uniform(-0.003, 0.003)
    time_c += random.uniform(-0.003, 0.003)

    # Create events as if from real nodes
    events = [
        {
            "node_id":    "node_a",
            "timestamp":  time_a,
            "node_lat":   NODE_A[0],
            "node_lon":   NODE_A[1],
            "confidence": 0.92,
            "drone_model": "DJI Phantom 4"
        },
        {
            "node_id":    "node_b",
            "timestamp":  time_b,
            "node_lat":   NODE_B[0],
            "node_lon":   NODE_B[1],
            "confidence": 0.89,
            "drone_model": "DJI Phantom 4"
        },
        {
            "node_id":    "node_c",
            "timestamp":  time_c,
            "node_lat":   NODE_C[0],
            "node_lon":   NODE_C[1],
            "confidence": 0.94,
            "drone_model": "DJI Phantom 4"
        },
    ]

    # Run TDOA
    result = calculate_drone_position(events)

    if result:
        # Calculate error vs true position
        error = haversine(
            result['lat'], result['lon'],
            DRONE_TRUE[0], DRONE_TRUE[1]
        )

        print(f"\nResults:")
        print(f"  True position:      {DRONE_TRUE}")
        print(f"  Calculated:         "
              f"({result['lat']:.6f}, "
              f"{result['lon']:.6f})")
        print(f"  Error:              {error:.1f} metres")
        print(f"  Confidence:         "
              f"{result['confidence']*100:.0f}%")

        if error < 25:
            print(f"\n  ✅ TDOA WORKING — "
                  f"error within acceptable range")
        else:
            print(f"\n  ⚠ Error higher than expected "
                  f"— check node positions")
    else:
        print("TDOA calculation returned None")


if __name__ == "__main__":
    test_trilateration()

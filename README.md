# AcoustiGuard

Passive drone detection and airspace monitoring system using crowd-sourced acoustic sensor networks and TDOA triangulation.

## What It Does
Detects drones by their acoustic fingerprint using a MobileNetV2 CNN, 
locates them to ±18m using TDOA trilateration, and displays their 
position on a live UTM dashboard.

## Tech Stack
Python · TensorFlow Lite · Flask · SocketIO · Leaflet.js · ESP32 · scipy

## Key Results
- 96% CNN accuracy on drone audio classification
- ±18m position accuracy via TDOA
- Under 3 second detection to dashboard latency
- ₹800 per IoT node — 100× cheaper than radar

## How to Run
1. pip install -r requirements.txt
2. python data/download_data.py
3. python model/train.py
4. cd tdoa && python server.py
5. python demo/simulate.py
6. Open http://localhost:5000

## Team
Team Clusters — POORVIKA SRINIVAS
SJB Institute of Technology
IISc FSID Competition
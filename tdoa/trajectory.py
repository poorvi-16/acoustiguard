# tdoa/trajectory.py
# Predicts drone flight path 30 seconds ahead
# Uses simple velocity-based prediction first,
# LSTM when enough data available

import numpy as np
from collections import deque

class TrajectoryPredictor:
    """
    Predicts where a drone will be in the next 30 seconds.
    Uses velocity extrapolation for small history,
    pattern matching for longer history.
    """

    def __init__(self, history_size=10):
        self.history = deque(maxlen=history_size)

    def add_position(self, lat, lon, timestamp):
        self.history.append({
            'lat': lat, 'lon': lon,
            'timestamp': timestamp
        })

    def predict(self, seconds_ahead=30, steps=10):
        """
        Returns list of predicted positions
        at evenly spaced time steps.
        """
        if len(self.history) < 2:
            return []

        positions = list(self.history)

        # Calculate average velocity vector
        # from recent positions
        lat_velocities = []
        lon_velocities = []

        for i in range(1, len(positions)):
            dt = positions[i]['timestamp'] - \
                 positions[i-1]['timestamp']
            if dt <= 0:
                continue
            dlat = positions[i]['lat'] - positions[i-1]['lat']
            dlon = positions[i]['lon'] - positions[i-1]['lon']
            lat_velocities.append(dlat / dt)
            lon_velocities.append(dlon / dt)

        if not lat_velocities:
            return []

        # Use recent velocity (last 3 readings weighted more)
        weights = np.exp(np.linspace(0, 1,
                        len(lat_velocities)))
        weights /= weights.sum()

        avg_lat_vel = np.average(lat_velocities, weights=weights)
        avg_lon_vel = np.average(lon_velocities, weights=weights)

        # Generate prediction points
        last = positions[-1]
        predictions = []
        time_step = seconds_ahead / steps

        for i in range(1, steps + 1):
            t = i * time_step
            predictions.append({
                'lat': last['lat'] + avg_lat_vel * t,
                'lon': last['lon'] + avg_lon_vel * t,
                'seconds_ahead': t,
            })

        return predictions


    def will_breach_geofence(self, fence_bounds, seconds=30):
        """
        Returns True if predicted path crosses geo-fence.
        fence_bounds: {'min_lat', 'max_lat',
                       'min_lon', 'max_lon'}
        """
        predictions = self.predict(seconds_ahead=seconds)

        for p in predictions:
            if (fence_bounds['min_lat'] <= p['lat'] <=
                fence_bounds['max_lat'] and
                fence_bounds['min_lon'] <= p['lon'] <=
                fence_bounds['max_lon']):
                return True, p['seconds_ahead']

        return False, None
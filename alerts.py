# alerts/alerts.py
# Sends real alerts when geo-fence is breached
# Email + SMS + webhook

import smtplib
import requests
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

class AlertSystem:
    """
    Multi-channel alert system for drone breaches.
    Supports Email, SMS (via Twilio/Fast2SMS),
    and webhook (for DGCA API integration).
    """

    def __init__(self, config):
        self.config    = config
        self.sent_log  = []
        self.cooldown  = {}  # prevent spam alerts

    def should_alert(self, drone_id, cooldown_sec=60):
        """Prevents same drone triggering multiple alerts."""
        import time
        now = time.time()
        if drone_id in self.cooldown:
            if now - self.cooldown[drone_id] < cooldown_sec:
                return False
        self.cooldown[drone_id] = now
        return True

    def send_email(self, subject, body):
        """Sends email alert to configured recipients."""
        try:
            msg = MIMEMultipart()
            msg['From']    = self.config['email_from']
            msg['To']      = self.config['email_to']
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(
                self.config['email_from'],
                self.config['email_password']
            )
            server.send_message(msg)
            server.quit()
            print(f"Email alert sent to {self.config['email_to']}")
            return True
        except Exception as e:
            print(f"Email failed: {e}")
            return False

    def send_sms(self, message):
        """
        Sends SMS via Fast2SMS (Indian SMS gateway).
        Free tier available — 50 SMS free per day.
        """
        try:
            url = "https://www.fast2sms.com/dev/bulkV2"
            payload = {
                "route":   "q",
                "message": message,
                "numbers": self.config['sms_number'],
            }
            headers = {
                "authorization": self.config['fast2sms_key']
            }
            r = requests.post(url, data=payload,
                            headers=headers, timeout=5)
            print(f"SMS sent: {r.json()}")
            return True
        except Exception as e:
            print(f"SMS failed: {e}")
            return False

    def send_webhook(self, position_data):
        """
        Sends position to any REST API endpoint.
        Can be used for DGCA UTM integration.
        """
        try:
            r = requests.post(
                self.config['webhook_url'],
                json=position_data,
                timeout=5
            )
            print(f"Webhook fired: {r.status_code}")
            return True
        except Exception as e:
            print(f"Webhook failed: {e}")
            return False

    def breach_alert(self, drone_id, lat, lon, confidence):
        """
        Main alert function — fires all channels
        when geo-fence is breached.
        """
        if not self.should_alert(drone_id):
            return

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        subject = f"⚠ DRONE BREACH — {drone_id} — AcoustiGuard"
        body = (
            f"DRONE GEO-FENCE BREACH DETECTED\n\n"
            f"Drone ID:    {drone_id}\n"
            f"Time:        {timestamp}\n"
            f"Position:    {lat:.5f}, {lon:.5f}\n"
            f"Confidence:  {confidence*100:.1f}%\n"
            f"Map:         "
            f"https://maps.google.com/?"
            f"q={lat},{lon}\n\n"
            f"Immediate response required.\n"
            f"— AcoustiGuard UTM System"
        )

        # Fire all channels
        if self.config.get('email_enabled'):
            self.send_email(subject, body)

        if self.config.get('sms_enabled'):
            self.send_sms(
                f"DRONE BREACH: {drone_id} at "
                f"{lat:.4f},{lon:.4f}. "
                f"Confidence: {confidence*100:.0f}%"
            )

        if self.config.get('webhook_enabled'):
            self.send_webhook({
                'drone_id':   drone_id,
                'lat':        lat,
                'lon':        lon,
                'confidence': confidence,
                'timestamp':  timestamp,
                'alert_type': 'GEOFENCE_BREACH',
            })

        self.sent_log.append({
            'drone_id': drone_id,
            'time':     timestamp,
            'lat':      lat,
            'lon':      lon,
        })
        print(f"All alerts fired for {drone_id}")


# Example config — fill in your details
ALERT_CONFIG = {
    'email_enabled':   True,
    'email_from':      'your_email@gmail.com',
    'email_to':        'authority@airport.in',
    'email_password':  'your_app_password',
    'sms_enabled':     True,
    'sms_number':      '9999999999',
    'fast2sms_key':    'YOUR_FAST2SMS_KEY',
    'webhook_enabled': False,
    'webhook_url':     'https://dgca-utm-api.example.com/alert',
}
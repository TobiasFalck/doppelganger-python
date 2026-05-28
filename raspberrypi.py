import sys
import time
import threading
from collections import deque
import json
import socket

import numpy as np
import RPi.GPIO as GPIO

sys.path.append("/home/pi/max30100")
import max30100
import I2C_LCD_driver


# -----------------------------
# LCD Setup
# -----------------------------
mylcd = I2C_LCD_driver.lcd()
mylcd.lcd_clear()
mylcd.lcd_display_string("Starting...", 1)
mylcd.lcd_display_string("Please wait", 2)


# -----------------------------
# Buzzer Setup
# -----------------------------
BUZZER_PIN = 4  # GPIO4 (BCM numbering)

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

# PWM for buzzer
buzzer = GPIO.PWM(BUZZER_PIN, 1000)
buzzer.stop()

heartbeat_active = False


def heartbeat_buzzer():
    """Play a calm heartbeat rhythm."""

    global heartbeat_active

    while heartbeat_active:

        # First beat (lub)
        buzzer.start(50)
        time.sleep(0.08)
        buzzer.stop()

        time.sleep(0.12)

        # Second beat (dub)
        buzzer.start(50)
        time.sleep(0.06)
        buzzer.stop()

        # Calm pause (~60 BPM feeling)
        time.sleep(0.75)


# -----------------------------
# Sensor Setup
# -----------------------------
mx30 = max30100.MAX30100()
mx30.set_mode(max30100.MODE_SPO2)

# Stronger LED signal
mx30.set_led_current(20.8, 20.8)

# Flush stale sensor FIFO data
for _ in range(50):
    mx30.read_sensor()
    time.sleep(0.01)


# -----------------------------
# Buffers
# -----------------------------
BUFFER_SIZE = 150

ir_buffer = deque(maxlen=BUFFER_SIZE)
red_buffer = deque(maxlen=BUFFER_SIZE)
timestamps = deque(maxlen=BUFFER_SIZE)

# Fresh startup
ir_buffer.clear()
red_buffer.clear()
timestamps.clear()

# Finger detection threshold
FINGER_THRESHOLD = 5000


# -----------------------------
# Startup Screen
# -----------------------------
print("Place your finger on the sensor.")
print("Keep still for best accuracy.")
print("Press Ctrl+C to stop.\n")

mylcd.lcd_clear()
mylcd.lcd_display_string("Place finger", 1)
mylcd.lcd_display_string("On sensor", 2)

# Short startup settle
time.sleep(2)

# Clear unstable startup samples
ir_buffer.clear()
red_buffer.clear()
timestamps.clear()

mylcd.lcd_clear()


# -----------------------------
# Peak Detection
# -----------------------------
def detect_peaks_simple(values, threshold=25):
    """Detect pulse peaks using local maxima."""
    if len(values) < 5:
        return []

    peaks = []

    for i in range(2, len(values) - 2):
        if (
            values[i] > values[i - 1]
            and values[i] > values[i - 2]
            and values[i] > values[i + 1]
            and values[i] > values[i + 2]
            and (
                values[i]
                - min(
                    values[i - 2],
                    values[i - 1],
                    values[i + 1],
                    values[i + 2],
                )
            )
            > threshold
        ):
            peaks.append(i)

    return peaks


# -----------------------------
# BPM Calculation
# -----------------------------
def calculate_bpm_from_buffer(
    values,
    timestamps_list,
):
    """Calculate BPM from pulse peaks."""

    if (
        len(values) < 40
        or len(timestamps_list) < 40
    ):
        return 0

    peaks_indices = detect_peaks_simple(
        values,
        threshold=20,
    )

    # Require at least 3 beats
    if len(peaks_indices) < 3:
        return 0

    peak_times = [
        timestamps_list[i]
        for i in peaks_indices
    ]

    intervals = []

    for i in range(1, len(peak_times)):
        interval = (
            peak_times[i]
            - peak_times[i - 1]
        )

        # Valid BPM range
        if 0.4 < interval < 1.5:
            intervals.append(interval)

    if not intervals:
        return 0

    avg_interval = (
        sum(intervals)
        / len(intervals)
    )

    bpm = int(60.0 / avg_interval)

    return bpm


# -----------------------------
# SpO2 Calculation
# -----------------------------
def calculate_spo2_improved(
    ir_vals,
    red_vals,
):
    """Calculate SpO2."""

    if (
        len(ir_vals) < 30
        or len(red_vals) < 30
    ):
        return 0

    ir_dc = np.mean(ir_vals)
    red_dc = np.mean(red_vals)

    if ir_dc == 0 or red_dc == 0:
        return 0

    # AC components
    ir_ac = np.std(ir_vals)
    red_ac = np.std(red_vals)

    # Reject weak signal
    if ir_ac < 5 or red_ac < 5:
        return 0

    # Ratio-of-ratios
    r = (
        (red_ac / red_dc)
        / (ir_ac / ir_dc)
    )

    spo2 = 110 - (25 * r)

    # Reject nonsense values
    if spo2 < 70 or spo2 > 100:
        return 0

    return int(spo2)


# -----------------------------
# Backend Socket Connection
# -----------------------------
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 5000
BACKEND_CONNECT_TIMEOUT = 5
BACKEND_SEND_TIMEOUT = 5
backend_socket = None
backend_lock = threading.Lock()


def connect_to_backend():
    """Open or restore the backend socket connection."""
    global backend_socket

    try:
        sock = socket.create_connection(
            (BACKEND_HOST, BACKEND_PORT),
            timeout=BACKEND_CONNECT_TIMEOUT,
        )
        sock.settimeout(BACKEND_SEND_TIMEOUT)
        backend_socket = sock
        print(
            f"Connected to backend {BACKEND_HOST}:{BACKEND_PORT}"
        )
        return True
    except OSError as exc:
        backend_socket = None
        print(
            f"Backend connection failed: {exc}"
        )
        return False


def close_backend_connection():
    """Close the backend socket cleanly."""
    global backend_socket

    if backend_socket is not None:
        try:
            backend_socket.close()
        except OSError:
            pass
        backend_socket = None


def send_measurement(
    bpm,
    spo2,
    raw_ir,
    raw_red,
    signal_quality,
):
    """Send a measurement payload to the backend."""
    with backend_lock:
        if backend_socket is None:
            if not connect_to_backend():
                return

        payload = {
            "timestamp": time.time(),
            "device_id": "borger-01",
            "bpm": bpm,
            "spo2": spo2,
            "raw_ir": raw_ir,
            "raw_red": raw_red,
            "signal_quality": signal_quality,
        }

        message = json.dumps(payload) + "\n"

        try:
            backend_socket.sendall(message.encode("utf-8"))
        except (BrokenPipeError, OSError) as exc:
            print(
                f"Backend send error: {exc}. Reconnecting."
            )
            close_backend_connection()
            if connect_to_backend():
                try:
                    backend_socket.sendall(
                        message.encode("utf-8")
                    )
                except (BrokenPipeError, OSError) as exc2:
                    print(
                        f"Backend send failed after reconnect: {exc2}"
                    )
                    close_backend_connection()


# -----------------------------
# Start Heartbeat Buzzer
# -----------------------------
heartbeat_active = True

heartbeat_thread = threading.Thread(
    target=heartbeat_buzzer,
    daemon=True,
)

heartbeat_thread.start()
connect_to_backend()


# -----------------------------
# Main Loop
# -----------------------------
try:
    last_print_time = time.time()

    while True:
        mx30.read_sensor()

        current_ir = mx30.ir
        current_red = mx30.red
        current_time = time.time()

        # -----------------------------
        # Finger Detection
        # -----------------------------
        if current_ir < FINGER_THRESHOLD:

            # Clear stale readings
            ir_buffer.clear()
            red_buffer.clear()
            timestamps.clear()

            print(
                f"No finger | "
                f"IR: {current_ir}"
            )

            mylcd.lcd_display_string(
                "No finger     ",
                1,
            )

            mylcd.lcd_display_string(
                "Place finger  ",
                2,
            )

            time.sleep(0.1)
            continue

        # -----------------------------
        # Store Sensor Data
        # -----------------------------
        ir_buffer.append(current_ir)
        red_buffer.append(current_red)
        timestamps.append(current_time)

        # -----------------------------
        # Stabilization Phase
        # -----------------------------
        if len(ir_buffer) < 40:

            mylcd.lcd_display_string(
                "Hold still... ",
                1,
            )

            mylcd.lcd_display_string(
                f"{len(ir_buffer):2d}/40",
                2,
            )

            time.sleep(0.02)
            continue

        # -----------------------------
        # Calculate Every 2 Seconds
        # -----------------------------
        if (
            current_time
            - last_print_time
            >= 2.0
        ):

            ir_list = list(ir_buffer)
            red_list = list(red_buffer)
            time_list = list(
                timestamps
            )

            bpm = (
                calculate_bpm_from_buffer(
                    ir_list,
                    time_list,
                )
            )

            spo2 = (
                calculate_spo2_improved(
                    ir_list,
                    red_list,
                )
            )

            # Signal quality
            ir_variation = (
                max(ir_list)
                - min(ir_list)
            )

            if ir_variation > 100:
                signal_quality = "Excellent"
            elif ir_variation > 50:
                signal_quality = "Good"
            elif ir_variation > 20:
                signal_quality = "Weak"
            else:
                signal_quality = "Poor"

            # Console output
            print(
                "━━━━━━━━━━━━━━━━━━━━━━"
            )

            print(
                f"Raw IR: "
                f"{current_ir:6d} | "
                f"Raw Red: "
                f"{current_red:6d}"
            )

            print(
                f"Signal: "
                f"{signal_quality}"
            )

            print(
                f"Heart Rate: "
                f"{bpm:3d} BPM | "
                f"SpO2: "
                f"{spo2:3d}%"
            )

            print(
                "━━━━━━━━━━━━━━━━━━━━━━\n"
            )

            # LCD output
            mylcd.lcd_display_string(
                f"Puls {bpm:3d} BPM   ",
                1,
            )

            mylcd.lcd_display_string(
                f"SAT {spo2:3d}%      ",
                2,
            )

            send_measurement(
                bpm=bpm,
                spo2=spo2,
                raw_ir=current_ir,
                raw_red=current_red,
                signal_quality=signal_quality,
            )

            last_print_time = (
                current_time
            )

            # Keep fresh data
            if len(ir_buffer) > 80:
                for _ in range(30):
                    ir_buffer.popleft()
                    red_buffer.popleft()
                    timestamps.popleft()

        # ~50Hz sample rate
        time.sleep(0.02)

except KeyboardInterrupt:
    print("\nMonitoring stopped")

    heartbeat_active = False
    buzzer.stop()

    mylcd.lcd_clear()
    GPIO.cleanup()
    close_backend_connection()

    print("Program exited cleanly")
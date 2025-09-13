import sys
import os
import time
import cv2
import serial
import serial.tools.list_ports

# Allow importing HandRobotController from the same folder
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from hand_detection import HandRobotController


def open_serial_port(preferred_ports, baudrate=115200, timeout=0.05):
    for port in preferred_ports:
        try:
            ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
            return ser
        except Exception:
            continue
    # Try auto-detect
    for p in serial.tools.list_ports.comports():
        try:
            ser = serial.Serial(p.device, baudrate=baudrate, timeout=timeout)
            return ser
        except Exception:
            continue
    return None


def clamp(v, lo=0, hi=180):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def map_deg_0_90_to_servo(v):
    # v expected 0..90 → 0..180
    if v < 0:
        v = 0
    if v > 90:
        v = 90
    return int(round((v / 90.0) * 180.0))


def _open_camera(preferred_id=0):
    for cam_id in [preferred_id, 1, 2, 0]:
        cap = cv2.VideoCapture(cam_id)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                return cap
            cap.release()
    return None


def main():
    controller = HandRobotController()

    uno_ports = ['/dev/ttyACM0', '/dev/ttyUSB0', 'COM3']
    master_ports = ['/dev/ttyACM1', '/dev/ttyUSB1', 'COM4']

    uno = open_serial_port(uno_ports)
    master = open_serial_port(master_ports)

    cap = _open_camera(0)
    if cap is None:
        print("Could not open camera.")
        return

    latest_raw = None
    last_send_time = 0.0
    send_interval = 0.05  # 20 Hz

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = controller.hands.process(rgb)

            if results.multi_hand_landmarks:
                # Use first detected hand
                hlms = results.multi_hand_landmarks[0]
                data = controller.process_hand_data(hlms.landmark)
                latest_raw = data

                # Servo values
                yaw_servo = int(data['orientation_servo']['yaw'])
                p1_servo = int(data['orientation_servo']['pitch1'])

                # Map finger pitches to p3/p4 servos
                p2_fingers = float(data['pitches_deg']['p2_fingers'])  # 0..90
                p3_tips = float(data['pitches_deg']['p3_tips'])        # 0..90
                p3_servo = map_deg_0_90_to_servo(p2_fingers)
                p4_servo = map_deg_0_90_to_servo(p3_tips)

                # Optional: include roll influence on p4 (commented)
                # p4_servo = clamp(int(90 + int(data['roll_int'])))

                now = time.time()
                if now - last_send_time >= send_interval:
                    last_send_time = now

                    if uno is not None:
                        try:
                            uno_string = f"yaw:{yaw_servo};p1:{p1_servo}\n"
                            uno.write(uno_string.encode())
                        except Exception:
                            pass

                    if master is not None:
                        try:
                            master_string = f"p3:{p3_servo};p4:{p4_servo}\n"
                            master.write(master_string.encode())
                        except Exception:
                            pass

                    # Read any reply from UNO for debugging
                    if uno is not None and uno.in_waiting:
                        try:
                            _ = uno.readline().decode(errors='ignore').strip()
                        except Exception:
                            pass

            # Minimal key handling without display
            if sys.platform.startswith('darwin'):
                # On macOS without window, waitKey has no effect; skip
                pass
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            time.sleep(0.005)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if uno is not None:
            try:
                uno.close()
            except Exception:
                pass
        if master is not None:
            try:
                master.close()
            except Exception:
                pass


if __name__ == '__main__':
    main()

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
    uno = open_serial_port(uno_ports)

    cap = _open_camera(0)
    if cap is None:
        print("Could not open camera.")
        return

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
                hlms = results.multi_hand_landmarks[0]
                data = controller.process_hand_data(hlms.landmark)

                # Servo values
                yaw_servo = int(data['orientation_servo']['yaw'])
                p1_servo = int(data['orientation_servo']['pitch1'])

                # Map finger pitches to p2/p3 servos
                p2_fingers = float(data['pitches_deg']['p2_fingers'])  # 0..90
                p3_tips = float(data['pitches_deg']['p3_tips'])        # 0..90
                p2_servo = map_deg_0_90_to_servo(p2_fingers)
                p3_servo = map_deg_0_90_to_servo(p3_tips)

                # Roll (use raw int degrees; clamp to [-90,90] then map to 0..180 if needed)
                roll_int = int(data['orientation_raw_deg']['roll'])
                roll_int = max(-90, min(90, roll_int))

                # Claw open/close -> cw (0.0 closed, 1.0 open)
                cw = 1.0 if data['claw']['open'] else 0.0

                now = time.time()
                if now - last_send_time >= send_interval:
                    last_send_time = now
                    if uno is not None:
                        try:
                            uno_string = (
                                f"yaw:{yaw_servo:.2f};"
                                f"p1:{p1_servo:.2f};"
                                f"p2:{p2_servo:.2f};"
                                f"p3:{p3_servo:.2f};"
                                f"roll:{float(roll_int):.2f};"
                                f"cw:{cw:.2f}\n"
                            )
                            print(uno_string.strip())
                            uno.write(uno_string.encode())
                        except Exception:
                            pass

            # Minimal key handling without display
            if sys.platform.startswith('darwin'):
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


if __name__ == '__main__':
    main()

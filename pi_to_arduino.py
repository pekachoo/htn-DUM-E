import serial
import time
import math

class IKSolver:
    def __init__(self, P1, P2, P3):
        self.P1 = P1
        self.P2 = P2
        self.P3 = P3

    def solve(self, x, y, phi, elbow='down'):
        """
        Inverse kinematics for a 3R planar arm.
        elbow='down' -> theta2 positive
        elbow='up'   -> theta2 negative
        """
        l1, l2, l3 = self.P1, self.P2, self.P3  # link lengths (cm)
        # wrist position
        xw = x - l3 * math.cos(phi)
        yw = y - l3 * math.sin(phi)
        r2 = xw*xw + yw*yw

        # cos(theta2)
        c2 = (r2 - l1*l1 - l2*l2) / (2*l1*l2)
        if c2 > 1 or c2 < -1:
            print("Target unreachable")
            return None  # unreachable

        s2_mag = abs(math.sqrt(1 - c2*c2))
        # sign choice: elbow-down = +, elbow-up = -
        s2 = s2_mag if elbow == 'down' else -s2_mag
        theta2 = math.atan2(s2, c2)

        # theta1
        k1 = l1 + l2 * c2
        k2 = l2 * s2
        theta1 = math.atan2(yw, xw) - math.atan2(k2, k1)

        # theta3
        theta3 = phi - theta1 - theta2

        # wrap to [-pi, pi]
        wrap = lambda a: (a + math.pi) % (2*math.pi) - math.pi

        return wrap(theta1), wrap(theta2), wrap(theta3)
    
def IK_to_servo_angles(angles):
    yaw, p1, p2, p3, roll, claw = angles
    # yaw
    yaw = -0.6556 * yaw + 105.0
    # p1
        ## restrain p1 angle
    if p1 > -90 and p1 < -35:
        p1 = -35
    if p1 < -90 and p1 > -149:
        p1 = -149
    if p1 < -90 and p1 >= -180:
        p1 += 360
    p1 = -0.0939 * p1 + 28
    # p2
    p2 = 0.6444 * p2 + 88
    # p3
    p3 = 90 - p3
    # roll
    roll = roll
    # claw
    claw = claw * 125
    return int(yaw), int(p1), int(p2), int(p3), int(roll), int(claw)
    
def sendTargets(angles, uno):
    yaw, p1, p2, p3, roll, claw = angles

    uno_string = f"yaw:{yaw:.2f};p1:{p1:.2f};p2:{p2:.2f};p3:{p3:.2f};roll:{roll:.2f};cw:{claw:.2f}\n"
    print(uno_string)
    uno.write(uno_string.encode())

def get_yaw_angle(x, y):
    angle = math.atan2(y, x)
    wrap = lambda a: (a + math.pi) % (2*math.pi) - math.pi
    return wrap(angle)

def projection(x, y, z):
    return math.sqrt(x**2 + y**2), z
    

def grab(ikSolver, x, y, phi, ser):
    t = time.time()
    while(time.time() - t <= 4):
        idle_angle = ikSolver.solve(math.sqrt(x**2 + y**2), 10, 300 * math.pi / 180, elbow='up')
        angles = (get_yaw_angle(x, y),) + idle_angle + (0, 1)  # open claw
        angles = tuple(int(math.degrees(a)) for a in angles)  # deg
        sendTargets(angles, ser)
        time.sleep(0.3)  # wait for arm to reach position
        
        #grab
        down_angle = ikSolver.solve(math.sqrt(x**2 + y**2), -5, phi * math.pi / 180, elbow='up')
        angles = (get_yaw_angle(x, y),) + down_angle + (0, 1)  # open claw
        angles = tuple(int(math.degrees(a)) for a in angles)  # deg
        sendTargets(angles, ser)
        time.sleep(0.3)  # wait for arm to reach position
        angles = (get_yaw_angle(x, y),) + down_angle + (0, 0)  # close claw
        angles = tuple(int(math.degrees(a)) for a in angles)  # deg
        sendTargets(angles, ser)
        time.sleep(0.3)  # wait for arm to reach position

        #lift
        angles = (get_yaw_angle(x, y),) + idle_angle + (0, 0)  # close claw
        angles = tuple(int(math.degrees(a)) for a in angles)  # deg
        sendTargets(angles, ser)
        time.sleep(0.3)  # wait for arm to reach position
        break

def move_to_idle_position(ikSolver, ser):
    t = time.time()
    while(time.time() - t <= 1.5):
        # SETTING TARGETS
        x, y, z = 4, 4, 22  # cm
        phi = 0 * math.pi / 180
        roll_angle, claw_open = 0, 1 

        # Calculate angles
        x_target, y_target = projection(x, y, z)
        pitch_angles = ikSolver.solve(x_target, y_target, phi, elbow='up')
        angles = (get_yaw_angle(x, y),) + pitch_angles + (roll_angle, claw_open)
        angles = tuple(int(math.degrees(a)) for a in angles)  # convert to degrees
        # sendTargets(angles, uno, master)
        print(angles)
        sendTargets(angles, ser)
        print("a")

        time.sleep(0.1)  # Control update rate

def move(ikSolver, x, y, z, phi, ser, claw_open=1, roll_angle=0, time_duration=2, elbow='up'):
    t = time.time()
    while(time.time() - t <= time_duration):
        # Calculate angles
        x_target, y_target = projection(x, y, z)
        pitch_angles = ikSolver.solve(x_target, y_target, phi, elbow=elbow)
        angles = (get_yaw_angle(x, y),) + pitch_angles + (roll_angle, claw_open)
        angles = tuple(int(math.degrees(a)) for a in angles)  # convert to degrees
        print(angles)
        print("b")
        sendTargets(angles, ser)

def wave_bye(ikSolver, ser):
    """Go to idle position, open claw, wave by oscillating L2 6 times"""
    move_to_idle_position(ikSolver, ser)
    time.sleep(2)
    for _ in range(3):
        move(ikSolver, -4, 0, 22, 135 * math.pi / 180, ser, claw_open=1, time_duration=1.5)
        move(ikSolver, 4, 0, 22, 45 * math.pi / 180, ser, claw_open=1, time_duration=1.5)

def shake_no(ikSolver):
    t=time.time()
    while(time.time() - t <= 1):
        move_to_idle_position(ikSolver)
        time.sleep(0.3)
        move(ikSolver, 0, 0, 22, 0, ser, claw_open=1, roll_angle=45, time_duration=1)
        time.sleep(0.3)
        move(ikSolver, 0, 0, 22, 0, ser, claw_open=1, roll_angle=135, time_duration=1)
        time.sleep(0.3)
        move(ikSolver, 0, 0, 22, 0, ser, claw_open=1, roll_angle=45, time_duration=1)
        time.sleep(0.3)
        move(ikSolver, 0, 0, 22, 0, ser, claw_open=1, roll_angle=135, time_duration=1)
        time.sleep(0.3)

def shake_yes(ikSolver):
    t=time.time()
    while(time.time() - t <= 1):
        move_to_idle_position(ikSolver)
        time.sleep(0.3)
        move(ikSolver, 0, 0, 22, 90, ser, claw_open=1, time_duration=1)
        time.sleep(0.3)
        move(ikSolver, 0, 0, 22, 180, ser, claw_open=1, time_duration=1)
        time.sleep(0.3)
        move(ikSolver, 0, 0, 22, 90, ser, claw_open=1, time_duration=1)
        time.sleep(0.3)
        move(ikSolver, 0, 0, 22, 180, ser, claw_open=1, time_duration=1)
        time.sleep(0.3)

def shake_hand(ikSolver):
    t=time.time()
    while(time.time() - t <= 1):
        move_to_idle_position(ikSolver)
        time.sleep(0.3)
        move(ikSolver, 0, 15, 10, 0, ser, claw_open=1, roll_angle=90, time_duration=1)
        time.sleep(2)
        move(ikSolver, 0, 15, 12, 0, ser, claw_open=1, roll_angle=90, time_duration=1)
        time.sleep(2)
        move(ikSolver, 0, 15, 10, 0, ser, claw_open=1, roll_angle=90, time_duration=1)
        time.sleep(2)
        move(ikSolver, 0, 15, 12, 0, ser, claw_open=1, roll_angle=90, time_duration=1)
        time.sleep(2)
        move(ikSolver, 0, 15, 10, 0, ser, claw_open=1, roll_angle=90, time_duration=1)

def hold(ikSolver):
    t=time.time()
    while(time.time() - t <= 1):
        move_to_idle_position(ikSolver)
        move(ikSolver, 4, 4, 22, 0, ser, claw_open=1, time_duration=1)
        time.sleep(5)
        move(ikSolver, 4, 4, 22, 0, ser, claw_open=0, time_duration=1)
        time.sleep(2)


if __name__ == "__main__":
    ser = serial.Serial("/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0",  9600, timeout=5)
    ikSolver = IKSolver(P1=12.0, P2=12.3, P3=8.0)
    time.sleep(2)

    try:
        # wave_bye(ikSolver, ser)
        move_to_idle_position(ikSolver, ser)
        # grab(ikSolver, 15, 15, 270, ser)

    except KeyboardInterrupt:
        print("Stopped by user.")

    # finally:
    #     uno.close()
    #     master.close()
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
        l1, l2, l3 = 12.0, 12.0, 6.0  # link lengths (cm)
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
    
def sendTargets(angles, uno, master):
    yaw, p1, p3, p4 = angles

    uno_string = f"yaw:{yaw:.2f};p1:{p1:.2f}\n"
    master_string = f"p3:{p3:.2f};p4:{p4:.2f}\n"

    uno.write(uno_string.encode())
    master.write(master_string.encode())

    if uno.in_waiting:
        print("Arduino:", uno.readline().decode().strip())

if __name__ == "__main__":

    ikSolver = IKSolver(P1=12, P2=12, P3=6)

    try:
        while True:
            x_target = 10
            y_target = 0
            phi = 270 * math.pi / 180

            angles = ikSolver.solve(x_target, y_target, phi, elbow='up')
            # sendTargets(angles, uno, master)
            print(angles)

            time.sleep(0.5)  # Control update rate

    except KeyboardInterrupt:
        print("Stopped by user.")

    # finally:
    #     uno.close()
    #     master.close()
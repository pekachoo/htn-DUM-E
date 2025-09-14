import serial
import time
ser = serial.Serial("/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0",  9600, timeout=1)

time.sleep(2)

while True:
    ser.write("yaw:30;p1:90;p2:30;p3:45;roll:30;cw:40".encode())
    time.sleep(2)
    ser.write("yaw:30;p1:15;p2:30;p3:45;roll:30;cw:40".encode())

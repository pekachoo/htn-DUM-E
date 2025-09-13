import serial
import time

ser = serial.Serial("/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0",  9600, timeout=1)
time.sleep(2)

while True:
    ser.write("ON".encode())
    time.sleep(2)
    ser.write("OFF".encode())
    time.sleep(2)
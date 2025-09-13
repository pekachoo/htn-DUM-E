#include <Servo.h>

// Create servo objects
Servo servoYaw;
Servo servoPitch1;
Servo servoPitch2;
Servo servoPitch3;
Servo servoClaw;

// Assign Arduino pins (adjust as needed)
const int YAW_PIN    = 3;
const int PITCH1_PIN = 5;
const int PITCH2_PIN = 6;
const int PITCH3_PIN = 9;
const int CLAW_PIN   = 10;

// Helper function to command all servos at once
void setServoPositions(int yaw, int p1, int p2, int p3, int claw) {
  servoYaw.write(yaw);
  servoPitch1.write(p1);
  servoPitch2.write(p2);
  servoPitch3.write(p3);
  servoClaw.write(claw);
}

void setup() {
  // Attach each servo to its pin
  servoYaw.attach(YAW_PIN);
  servoPitch1.attach(PITCH1_PIN);
  servoPitch2.attach(PITCH2_PIN);
  servoPitch3.attach(PITCH3_PIN);
  servoClaw.attach(CLAW_PIN);

  // Optional: set initial positions (degrees 0–180)
  servoYaw.write(90);
  servoPitch1.write(90);
  servoPitch2.write(90);
  servoPitch3.write(90);
  servoClaw.write(0);  // start with claw open
}

void loop() {
  // Example sequence: move each servo to a new position
  setServoPositions(120, 80, 100, 60, 45);  // yaw, p1, p2, p3, claw
  delay(2000);

  setServoPositions(60, 100, 80, 120, 0);
  delay(2000);
}



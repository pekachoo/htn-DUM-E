#include <Servo.h>

// Create servo objects
Servo servoYaw;
Servo servoPitch1;
Servo servoPitch2;
Servo servoPitch3;
Servo roll;
Servo servoClaw;

// Assign Arduino pins (adjust as needed)
const int YAW_PIN    = 6;
const int PITCH1_PIN = 7;
const int PITCH2_PIN = 8;
const int PITCH3_PIN = 9;
const int ROLL_PIN = 10;
const int CLAW_PIN = 11;

void deserializeAndSetServos(String data) {
  // Expected format: "yaw:30;p1:15;p2:30;p3:45;roll:30;claw:40"
  int yaw, p1, p2, p3, roll_pos, claw;
  sscanf(data.c_str(), "yaw:%d;p1:%d;p2:%d;p3:%d;roll:%d;claw:%d", &yaw, &p1, &p2, &p3, &roll_pos, &claw);
  setServoPositions(yaw, p1, p2, p3, roll_pos, claw);
}

struct ServoAngles {
  int yaw;
  int p1;
  int p2;
  int p3;
  int roll;
  int claw;
};

ServoAngles IK_to_servo_angles(float yaw, float p1, float p2, float p3, float roll, float claw) {
  // yaw
  yaw = -0.6889f * yaw + 124.0f;

  // p1 restraints
  if (p1 > -90 && p1 < -35) {
    p1 = -35;
  }
  if (p1 < -90 && p1 > -149) {
    p1 = -149;
  }
  if (p1 < -90 && p1 >= -180) {
    p1 += 360;
  }
  p1 = -0.0939f * p1 + 28.0f;

  // p2
  p2 = 0.6444f * p2 + 88.0f;

  // p3
  p3 = 90.0f - p3;

  // roll stays unchanged
  // claw scaled
  claw = claw * 125.0f;

  ServoAngles result;
  result.yaw  = (int)yaw;
  result.p1   = (int)p1;
  result.p2   = (int)p2;
  result.p3   = (int)p3;
  result.roll = (int)roll;
  result.claw = (int)claw;

  return result;
}


// Helper function to command all servos at once
void setServoPositions(int yaw, int p1, int p2, int p3, int roll_pos, int claw) {
  servoYaw.write(yaw);
  delay(5000);
  servoPitch1.write(p1);
  delay(5000);
  servoPitch2.write(p2);
  delay(5000);
  servoPitch3.write(p3);
  delay(5000);
  roll.write(roll_pos);
  delay(5000);
  servoClaw.write(claw);
  delay(5000);
}

void setup() {
  // Attach each servo to its pin
  servoYaw.attach(YAW_PIN);
  servoPitch1.attach(PITCH1_PIN);
  servoPitch2.attach(PITCH2_PIN);
  servoPitch3.attach(PITCH3_PIN);
  roll.attach(ROLL_PIN);
  servoClaw.attach(CLAW_PIN);
}

void loop() {
  // Example sequence: move each servo to a new position
  // Test 1: does each servo go to its position?
  result = IK_to_servo_angles(0,0,0,0,0,0);  // yaw, p1, p2, p3, roll, claw
  setServoPositions(result.yaw, result.p1, result.p2, result.p3, result.roll, result.claw);
  delay(2000);
}



#include <Servo.h>

Servo myServo;

void setup() {
  myServo.attach(9);  // Connect the servo signal wire to pin 9
}

void loop() {
  // Move from 0 to 180 degrees
  // for (int pos = 0; pos <= 180; pos++) {
  //   myServo.write(pos);
  //   delay(15);           // small pause so the servo can reach the position
  // }
  myServo.write(0);

  delay(5000);  // wait half a second

  myServo.write(90);

  // Move back to 0 degrees
  // for (int pos = 180; pos >= 0; pos--) {
  //   myServo.write(pos);
  //   delay(15);
  // }

  delay(1000);
}

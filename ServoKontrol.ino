#include <Servo.h>

Servo motorpitch, motoryaw, motorentry1 ,motorentry2;
int serialData,motor1Data, motor2Data,distance;
const int trigPin = 6;
const int echoPin = 7;
long duration;
void setup() {
  motorpitch.attach(2);
  motoryaw.attach(3);
  motorentry1.attach(5);
  motorentry2.attach(4);
  Serial.begin(9600);
  motorpitch.write(60);
  motoryaw.write(60);
  motorentry1.write(3);
  motorentry2.write(120);
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);

}
void loop() {
    digitalWrite(trigPin, LOW);
    delayMicroseconds(2);
    digitalWrite(trigPin, HIGH);
    delayMicroseconds(10);
    digitalWrite(trigPin, LOW);
    
    duration = pulseIn(echoPin, HIGH);
    
    distance = duration*0.034/2;
    Serial.println(String(distance));
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n'); 
    if(input == "home"){
      motorentry1.write(3);
      motorpitch.write(60);
      motoryaw.write(60);
      motorentry2.write(120);
    }
    else{
    int spaceIndex = input.indexOf(' ');
    int entryindex = input.indexOf('-');
    
    if (spaceIndex != -1) {
      String motorpitchstate = input.substring(0, spaceIndex);
      String motoryawstate = input.substring(spaceIndex + 1);
      String motorentrystate = input.substring(entryindex + 1);

      motorpitch.write(motorpitchstate.toInt());
      motoryaw.write(motoryawstate.toInt());
      delay(1000);
      if(motorentrystate.toInt() == 1){
          motorentry1.write(140);
          motorentry2.write(0);
      }else{
          motorentry1.write(3);
          motorentry2.write(120);
      }
    }
  }
  }

  delay(100);
}

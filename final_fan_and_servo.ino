#include <Stepper.h>

const int stepsPerRevolution = 2048 ;
int inByte = 9;
int angle = 0;
int base_pin = 6;
Stepper myStepper(stepsPerRevolution, 11,9,10,8);            

void handcontrol_mode();
void test_mode();

void setup() {
  Serial.begin(115200); 
  myStepper.setSpeed(5);
  pinMode(base_pin,OUTPUT);
  analogWrite(base_pin,0);
}

void loop() {
  test_mode();
}

void handcontrol_mode(){
    if (Serial.available()>0){  
      inByte= Serial.read();
      switch(int(inByte)){
        case 7:
          myStepper.step(15);
          analogWrite(base_pin,255);
          angle+=15;
          break;
        case 8:
          myStepper.step(-15);
          analogWrite(base_pin,255);
          angle-=15;
          break;
        case 5:
          myStepper.step(-angle);
          analogWrite(base_pin,255);
          angle = 0;
          break;
        case 0:
          analogWrite(base_pin,0);
          break;
      }
      Serial.println(int(inByte));
      delay(5);
      inByte = 9;
      }
}

void test_mode(){
  analogWrite(base_pin,255);
  myStepper.step(300);
  delay(100);
  myStepper.step(-300);
  delay(100);
  }


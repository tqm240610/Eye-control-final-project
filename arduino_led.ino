/*
  arduino_led.ino
  Nhan lenh tu may tinh qua Serial (USB) de dieu khien hai den LED.

  Day noi:
    LED xanh (+) -> dien tro 220-330 ohm -> chan D8
    LED xanh (-) -> GND
    LED do (+)   -> dien tro 220-330 ohm -> chan D9
    LED do (-)   -> GND

  Lenh nhan duoc tu Serial (ket thuc bang newline '\n'):
    "1" hoac "ON" -> bat den xanh (tuong thich code cu)
    "0" hoac "OFF" -> tat ca hai den
    "green" -> chi bat den xanh; "red" -> chi bat den do; "none" -> tat ca hai

  Toc do Serial: 9600 baud, phai khop voi Python (pyserial) o may tinh.
*/

const int GREEN_LED_PIN = 8;
const int RED_LED_PIN = 9;

void setLeds(bool greenOn, bool redOn) {
  digitalWrite(GREEN_LED_PIN, greenOn ? HIGH : LOW);
  digitalWrite(RED_LED_PIN, redOn ? HIGH : LOW);
}

void setup() {
  pinMode(GREEN_LED_PIN, OUTPUT);
  pinMode(RED_LED_PIN, OUTPUT);
  setLeds(false, false);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    String lenh = Serial.readStringUntil('\n');
    lenh.trim();
    if (lenh == "1" || lenh == "ON") {
      setLeds(true, false);
      Serial.println("LED:GREEN");
    } else if (lenh == "0" || lenh == "OFF") {
      setLeds(false, false);
      Serial.println("LED:NONE");
    } else if (lenh == "green") {
      setLeds(true, false);
      Serial.println("LED:GREEN");
    } else if (lenh == "red") {
      setLeds(false, true);
      Serial.println("LED:RED");
    } else if (lenh == "none") {
      setLeds(false, false);
      Serial.println("LED:NONE");
    }
  }
}

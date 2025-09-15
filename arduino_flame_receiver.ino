/*
Arduino code for Flame Detection Communication
Receives flame position from Raspberry Pi and responds accordingly
*/

void setup() {
  Serial.begin(9600);
  Serial.println("Arduino Flame Detection Ready");
}

void loop() {
  if (Serial.available()) {
    String position = Serial.readStringUntil('\n');
    position.trim();
    
    if (position == "LEFT") {
      Serial.println("Flame detected on LEFT side");
      // Add your LEFT action here (LED, servo, etc.)
    }
    else if (position == "RIGHT") {
      Serial.println("Flame detected on RIGHT side");
      // Add your RIGHT action here (LED, servo, etc.)
    }
    else if (position == "CENTER") {
      Serial.println("Flame detected in CENTER");
      // Add your CENTER action here (LED, servo, etc.)
    }
    else if (position == "NONE") {
      Serial.println("No flame detected");
      // Add your NONE action here (turn off LEDs, etc.)
    }
  }
  
  delay(50); // Small delay for stability
}

#!/usr/bin/env python3
"""
Simple Serial Communication Test
Test Arduino connection and data sending
"""

import serial
import serial.tools.list_ports
import time

def test_serial_communication():
    """Test simple serial communication"""
    print("=== Serial Communication Test ===")
    
    # List available ports
    ports = list(serial.tools.list_ports.comports())
    print(f"Available ports: {len(ports)}")
    
    for i, port in enumerate(ports):
        print(f"{i}: {port.device} - {port.description}")
    
    if not ports:
        print("No serial ports found!")
        return
    
    # Try to connect to first port
    try:
        arduino = serial.Serial(
            port=ports[0].device,
            baudrate=9600,
            timeout=1
        )
        time.sleep(2)  # Allow connection to establish
        print(f"Connected to: {ports[0].device}")
        
        # Send test data
        test_messages = ["LEFT", "RIGHT", "CENTER", "NONE"]
        
        for i in range(10):  # Send 10 test messages
            message = test_messages[i % 4]
            arduino.write(f"{message}\n".encode('utf-8'))
            print(f"Sent: {message}")
            
            # Try to read response (optional)
            try:
                if arduino.in_waiting > 0:
                    response = arduino.readline().decode('utf-8').strip()
                    print(f"Received: {response}")
            except:
                pass
            
            time.sleep(1)  # Wait 1 second between messages
        
        arduino.close()
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Make sure Arduino is connected and not used by other applications")

if __name__ == "__main__":
    test_serial_communication()

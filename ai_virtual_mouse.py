import cv2
from cvzone.HandTrackingModule import HandDetector
import mouse
import threading
import numpy as np
import time

# Parameters
frameR = 100  # Frame boundary
cam_w, cam_h = 640, 480

# Initialize camera
cap = cv2.VideoCapture(0)  
cap.set(3, cam_w)
cap.set(4, cam_h)

# Hand detector from cvzone
detector = HandDetector(detectionCon=0.9, maxHands=1)

# Delays for clicks
l_delay = 0
r_delay = 0
double_delay = 0

# Functions to manage click delays with threading
def l_clk_delay():
    global l_delay
    time.sleep(1)
    l_delay = 0

def r_clk_delay():
    global r_delay
    time.sleep(1)
    r_delay = 0

def double_clk_delay():
    global double_delay
    time.sleep(2)
    double_delay = 0

# Main loop
while True:
    success, img = cap.read()
    if success:
        img = cv2.flip(img, 1)  # Flip image for mirror view

        # Detect hands and landmarks
        hands, img = detector.findHands(img, flipType=False)
        cv2.rectangle(img, (frameR, frameR), (cam_w - frameR, cam_h - frameR), (255, 0, 255), 2)  # Draw frame border

        if hands:
            # Get landmarks for the first hand
            lmlist = hands[0]['lmList']
            ind_x, ind_y = lmlist[8][0], lmlist[8][1]  # Index finger
            mid_x, mid_y = lmlist[12][0], lmlist[12][1]  # Middle finger

            # Draw circles at index and middle finger tips
            cv2.circle(img, (ind_x, ind_y), 5, (0, 255, 255), 2)
            cv2.circle(img, (mid_x, mid_y), 5, (0, 255, 255), 2)

            # Get which fingers are up
            fingers = detector.fingersUp(hands[0])

            # Mouse movement (index finger up, thumb up)
            if fingers[1] == 1 and fingers[2] == 0 and fingers[0] == 1:
                conv_x = int(np.interp(ind_x, (frameR, cam_w - frameR), (0, 1536)))
                conv_y = int(np.interp(ind_y, (frameR, cam_h - frameR), (0, 864)))
                mouse.move(conv_x, conv_y)
                print(f"Mouse moved to: {conv_x}, {conv_y}")

            # Mouse button clicks (index and middle fingers up)
            if fingers[1] == 1 and fingers[2] == 1 and fingers[0] == 1:
                if abs(ind_x - mid_x) < 25:  # If index and middle fingers are close
                    # Left click (pinky down)
                    if fingers[4] == 0 and l_delay == 0:
                        mouse.click(button="left")
                        l_delay = 1
                        threading.Thread(target=l_clk_delay).start()

                    # Right click (pinky up)
                    if fingers[4] == 1 and r_delay == 0:
                        mouse.click(button="right")
                        r_delay = 1
                        threading.Thread(target=r_clk_delay).start()

            # Scrolling
            if fingers[1] == 1 and fingers[2] == 1 and fingers[0] == 0:
                if abs(ind_x - mid_x) < 25:
                    if fingers[4] == 0:  # Scroll down
                        mouse.wheel(delta=-1)
                    elif fingers[4] == 1:  # Scroll up
                        mouse.wheel(delta=1)

            # Double click (index finger up, all others down)
            if fingers[1] == 1 and all(f == 0 for f in [fingers[2], fingers[0], fingers[3], fingers[4]]) and double_delay == 0:
                mouse.double_click(button="left")
                double_delay = 1
                threading.Thread(target=double_clk_delay).start()

        # Display the camera feed
        cv2.imshow("Camera Feed", img)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()

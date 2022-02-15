import cv2
import numpy as np
from unified_detector import Fingertips
from hand_detector.detector import SOLO, YOLO
import math
import time
import pyautogui


pyautogui.FAILSAFE = False      #auto exit program turned off
SCREEN_X, SCREEN_Y = pyautogui.size()   # current screen resolution width and height
CLICK = CLICK_MESSAGE = MOVEMENT_START = M_START = None   # set null value


hand_detection_method = 'yolo'  # Do not use solo for video feed

if hand_detection_method is 'solo':
    hand = SOLO(weights='weights\solo.h5', threshold=0.8)
elif hand_detection_method is 'yolo':
    hand = YOLO(weights='weights\yolo.h5', threshold=0.8)
else:
    assert False, "'" + hand_detection_method + "' hand detection does not exist. use either 'solo' or 'yolo' as hand detection method"

fingertips = Fingertips(weights='weights/classes8.h5')

cam = cv2.VideoCapture(0)   #captures video from webcam
print('AirCtrl Demo')

# Identifies finger

while True:
    ret, image = cam.read()
    CAMERA_X, CAMERA_Y, channels = image.shape 
    if ret is False:
        break

    # hand detection
    tl, br = hand.detect(image=image)
    if tl and br is not None:
        cropped_image = image[tl[1]:br[1], tl[0]: br[0]]
        height, width, _ = cropped_image.shape

        # gesture classification and fingertips regression
        prob, pos = fingertips.classify(image=cropped_image)
        pos = np.mean(pos, 0)

        # post-processing
        prob = np.asarray([(p >= 0.5) * 1.0 for p in prob])
        for i in range(0, len(pos), 2):
            pos[i] = pos[i] * width + tl[0]
            pos[i + 1] = pos[i + 1] * height + tl[1]
            x=pos.astype(pos[1])
            y=pos.astype(pos[2])

        # drawing
        index = 0
        color = [(15, 15, 240), (15, 240, 155), (240, 155, 15), (240, 15, 155), (240, 15, 240)]
        image = cv2.rectangle(image, (tl[0], tl[1]), (br[0], br[1]), (235, 26, 158), 2)
        for c, p in enumerate(prob):
            if p > 0.5:
                display_c = int(c)
                image = cv2.circle(image, (int(pos[index]), int(pos[index + 1])), radius=12,
                                   color=color[c], thickness=-2)
                #cv2.putText(image, str(c), (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 3, 3)
                
 
                  # Mouse control
                
                # print("C: " + str(c), str(p))
               # if c is not None:    #thumb and index
                x = pos[0]
                y = pos[1]
                display_x = x
                display_y = y
                print(M_START)
                print(MOVEMENT_START)
                M_START = (display_x, display_y)
                if c:
                    print("C1: " + str(c), str(p))
                    if MOVEMENT_START is not None:
                        print("C2: " + str(c), str(p))
                        M_START = (display_x, display_y)
                        display_x = display_x - MOVEMENT_START[0]
                        display_y = display_y - MOVEMENT_START[1]
                        display_x = display_x * (SCREEN_X / CAMERA_X)
                        display_y = display_y * (SCREEN_Y / CAMERA_Y)
                        MOVEMENT_START = M_START
                        print("X: " + str(display_x) + " Y: " + str(display_y) + " C: " + str(c) + " diffDisplay_X" + str(MOVEMENT_START[0]))
                        pyautogui.moveRel(display_x, display_y)     # move mouse relative to its current position (2 defects)
                        # if c == 4 and CLICK is None:
                        #     CLICK = time.time()
                        #     pyautogui.click()   # left clicks when pinky finger present
                        #     CLICK_MESSAGE = "LEFT CLICK"
                        # elif c == 0 and CLICK is None:
                        #     CLICK = time.time()
                        #     pyautogui.rightClick()      # Right click when thumb present
                        #     CLICK_MESSAGE = "RIGHT CLICK"

                    elif MOVEMENT_START is None:
                                MOVEMENT_START = (M_START)
            else:
                MOVEMENT_START = None

            if CLICK is not None:
                cv2.putText(image, str(c) + CLICK_MESSAGE, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, 3)
                if CLICK < time.time():
                    CLICK = None
            index = index + 2


# ESC key to stop program
    if cv2.waitKey(1) & 0xff == 27:
        break

    # display image
    cv2.imshow('AirCtrl Demo', image)
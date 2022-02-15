import cv2
import numpy as np
from hand_detector.detector import YOLO
from unified_detector import Fingertips

hand = YOLO(weights='weights/yolo.h5', threshold=0.8)
fingertips = Fingertips(weights='weights/classes8.h5')

image = cv2.imread('data/sample.jpg')
tl, br = hand.detect(image=image)
if tl or br is not None:
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

    # drawing
    index = 0
    color = [(15, 15, 240), (15, 240, 155), (240, 155, 15), (240, 15, 155), (240, 15, 240)]
    image = cv2.rectangle(image, (tl[0], tl[1]), (br[0], br[1]), (235, 26, 158), 2)
    for c, p in enumerate(prob):
        if p > 0.5:
            image = cv2.circle(image, (int(pos[index]), int(pos[index + 1])), radius=12, color=color[c], thickness=-2)
        index = index + 2

    # display image
    cv2.imshow('AirCtrl Demo', image)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

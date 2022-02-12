import numpy as np
from preprocess.finger_flag import Finger


def class_finder(prob):
    cls = None
    classes = [0, 1, 2, 3, 4, 5, 6, 7]

    if np.array_equal(prob, np.array([0, 1, 0, 0, 0])):
        cls = classes[0]
    elif np.array_equal(prob, np.array([0, 1, 1, 0, 0])):
        cls = classes[1]
    elif np.array_equal(prob, np.array([0, 1, 1, 1, 0])):
        cls = classes[2]
    elif np.array_equal(prob, np.array([0, 1, 1, 1, 1])):
        cls = classes[3]
    elif np.array_equal(prob, np.array([1, 1, 1, 1, 1])):
        cls = classes[4]
    elif np.array_equal(prob, np.array([1, 0, 0, 0, 1])):
        cls = classes[5]
    elif np.array_equal(prob, np.array([1, 1, 0, 0, 1])):
        cls = classes[6]
    elif np.array_equal(prob, np.array([1, 1, 0, 0, 0])):
        cls = classes[7]
    return cls


def find_folder(image_name):
    name = image_name.split('_')
    if 'One' in name:
        folder_name = 'SingleOne'
    elif 'Two' in name:
        folder_name = 'SingleTwo'
    elif 'Three' in name:
        folder_name = 'SingleThree'
    elif 'Four' in name:
        folder_name = 'SingleFour'
    elif 'Five' in name:
        folder_name = 'SingleFive'
    elif 'Six' in name:
        folder_name = 'SingleSix'
    elif 'Seven' in name:
        folder_name = 'SingleSeven'
    elif 'Eight' in name:
        folder_name = 'SingleEight'
    elif 'Nine' in name:
        folder_name = 'SingleNine'
    elif 'Good' in name:
        folder_name = 'SingleGood'
    elif 'Bad' in name:
        folder_name = 'SingleBad'
    else:
        folder_name = 'SingleNone'
    return folder_name


def finger_type(image_name):
    name = image_name.split('_')
    finger = None
    if 'One' in name:
        finger = Finger().SingleOne()
    elif 'Two' in name:
        finger = Finger().SingleTwo()
    elif 'Three' in name:
        finger = Finger().SingleThree()
    elif 'Four' in name:
        finger = Finger().SingleFour()
    elif 'Five' in name:
        finger = Finger().SingleFive()
    elif 'Six' in name:
        finger = Finger().SingleSix()
    elif 'Seven' in name:
        finger = Finger().SingleSeven()
    elif 'Eight' in name:
        finger = Finger().SingleEight()
    return finger

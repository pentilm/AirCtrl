import cv2
import numpy as np
from preprocess.finder import find_folder, finger_type


def label_generator_testset(directory, image_name, type='Test'):
    """
    Generates ground truths data
    :returns
    original_image: test image
    tl: top left coordinate of the hand position
    cropped_image: cropped hand image
    ground_truths: list of probability and key points position
    """

    folder_name = find_folder(image_name=image_name)
    image = cv2.imread(directory + folder_name + type + '/' + image_name)
    original_image = cv2.resize(image, (640, 480))

    """ Reading corresponding keypoints """
    file = open(directory + 'label/' + folder_name + '.txt')
    lines = file.readlines()
    file.close()

    label = []
    for line in lines:
        line = line.strip().split()
        name = line[0].split('/')[3]
        if image_name == name:
            label = line[1:]
            break

    for i in range(0, len(label), 2):
        label[i] = float(label[i]) * 640
        label[i + 1] = float(label[i + 1]) * 480

    # Top-left and bottom-right coordinates
    tl = (label[0], label[1])
    x1 = int(label[0])
    y1 = int(label[1])
    x2 = int(label[2])
    y2 = int(label[3])

    # Finger key points
    keys = label[4:]

    # Correcting bounding box for outside of the image
    x1 = x1 if x1 > 0 else 0
    y1 = y1 if y1 > 0 else 0
    x2 = x2 if x2 < 640 else 640
    y2 = y2 if y2 < 480 else 480

    # Adding additional boundary if alpha!= 0
    alpha = 0
    y1 = y1 - alpha if y1 - alpha > 0 else 0
    x1 = x1 - alpha if x1 - alpha > 0 else 0
    if y1 == 0:
        alpha = 0
    if x1 == 0:
        alpha = 0

    cropped_image = original_image[y1:y2 + alpha, x1:x2 + alpha]
    cols, rows, _ = cropped_image.shape

    # new keypoints for cropped image
    label = []
    for i in range(0, len(keys), 4):
        x = keys[i]
        y = keys[i + 1]
        label.append(x)
        label.append(y)

    prob = finger_type(image_name=image_name)
    t, i, m, r, p = prob

    count = 0
    keypoints = []
    if t == 1:
        keypoints.append(label[count])
        keypoints.append(label[count + 1])
        count = count + 2
    else:
        keypoints.append(0.0)
        keypoints.append(0.0)
    if i == 1:
        keypoints.append(label[count])
        keypoints.append(label[count + 1])
        count = count + 2
    else:
        keypoints.append(0.0)
        keypoints.append(0.0)
    if m == 1:
        keypoints.append(label[count])
        keypoints.append(label[count + 1])
        count = count + 2
    else:
        keypoints.append(0.0)
        keypoints.append(0.0)
    if r == 1:
        keypoints.append(label[count])
        keypoints.append(label[count + 1])
        count = count + 2
    else:
        keypoints.append(0.0)
        keypoints.append(0.0)
    if p == 1:
        keypoints.append(label[count])
        keypoints.append(label[count + 1])
    else:
        keypoints.append(0.0)
        keypoints.append(0.0)

    prob = np.asarray(prob) * 1.0
    keypoints = np.asarray(keypoints)
    ground_truths = [prob, keypoints]
    return original_image, tl, cropped_image, ground_truths


if __name__ == '__main__':
    img_name = 'ChuangyeguBusstop_Single_Six_color_18.jpg'
    dir = '../../EgoGesture Dataset/'
    original_image, tl, cropped_image, ground_truths = label_generator_testset(directory=dir,
                                                                               image_name=img_name,
                                                                               type='Test')

    prob = ground_truths[0]
    key = ground_truths[1]
    # print(prob)
    # print(key)

    index = 0
    color = [(120, 20, 240), (240, 55, 210), (240, 55, 140), (240, 75, 55), (170, 240, 55)]
    for c, p in enumerate(prob):
        if p > 0.5:
            original_image = cv2.circle(original_image, (int(key[index]), int(key[index + 1])), radius=14,
                                        color=color[c], thickness=-2)
        index = index + 2

    cv2.imshow('Test Image with Ground Truths', original_image)
    cv2.waitKey(0)

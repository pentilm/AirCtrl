import cv2
import numpy as np
from preprocess.finder import find_folder, finger_type


def label_generator(directory, image_name, type=''):
    """
    Generate input image and output label of CNN
    directory: dataset directory
    image_name: name of the image file
    type: type of image, if train then '', if valid then 'Valid', if test then 'Test'
    """

    folder_name = find_folder(image_name=image_name)
    image = cv2.imread(directory + folder_name + type + '/' + image_name)
    image = cv2.resize(image, (640, 480))

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

    # Bounding box
    x1 = int(label[0])
    y1 = int(label[1])
    x2 = int(label[2])
    y2 = int(label[3])

    alpha = 0
    x1 = x1 - alpha if x1 - alpha > 0 else 0
    y1 = y1 - alpha if y1 - alpha > 0 else 0

    image = image[y1:y2 + alpha, x1:x2 + alpha]
    cols, rows, _ = image.shape
    image = cv2.resize(image, (128, 128))

    # keypoints
    keys = label[4:]
    keys = [float(k) for k in keys]

    # new keypoints
    label = []
    for i in range(0, len(keys), 4):
        x = (keys[i] - x1) * 128 / rows
        y = (keys[i + 1] - y1) * 128 / cols
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
    return image, prob, keypoints


if __name__ == '__main__':
    img_name = 'ChuangyeguFirstfloor_Single_Four_color_335.jpg'
    dir = '../../EgoGesture Dataset/'
    img, prob, keys = label_generator(directory=dir, image_name=img_name, type='Test')
    print(prob)
    print(keys)

    # drawing keypoints
    index = 0
    color = [(120, 20, 240), (240, 55, 210), (240, 55, 140), (240, 75, 55), (170, 240, 55)]
    for c, p in enumerate(prob):
        if p > 0.5:
            img = cv2.circle(img, (int(keys[index]), int(keys[index + 1])), radius=5, color=color[c], thickness=-2)
        index = index + 2
    cv2.imshow('', img)
    # cv2.imwrite('image.jpg', img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

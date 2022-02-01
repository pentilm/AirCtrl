import cv2
import numpy as np
import matplotlib.pyplot as plt
from preprocess.finder import find_folder
from hand_detector.yolo.utils.utils import visualize
from hand_detector.yolo.preprocess.yolo_flag import Flag

f = Flag()

grid = f.grid
grid_size = f.grid_size
target_size = f.target_size
threshold = f.threshold


def label_generator(directory, image_name, type=''):
    folder_name = find_folder(image_name)
    image = plt.imread(directory + folder_name + type + '/' + image_name)
    image = cv2.resize(image, (target_size, target_size))

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

    """ bbox: top-left and bottom-right coordinate of the bounding box """
    label = label[0:4]
    bbox = [float(element) * target_size for element in label]
    bbox = np.array(bbox)
    return image, bbox


def bbox_to_grid(bbox):
    output = np.zeros(shape=(grid, grid, 5))
    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    i, j = int(np.floor(center[0] / grid_size)), int(np.floor(center[1] / grid_size))
    i = i if i < f.grid else f.grid - 1
    j = j if j < f.grid else f.grid - 1
    output[i, j, 0] = 1
    output[i, j, 1:] = bbox / f.target_size
    return output


if __name__ == '__main__':
    img_name = 'ChuangyeguBusstop_Single_Five_color_26.jpg'
    dir = '../../../../EgoGesture Dataset/'
    img, box = label_generator(directory=dir, image_name=img_name)
    yolo_out = bbox_to_grid(box)
    visualize(img, yolo_out, title='', RGB2BGR=True)

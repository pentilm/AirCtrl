import numpy as np
from preprocess.iou import iou
from preprocess.finder import find_folder
from hand_detector.solo.preprocess.solo_flag import Flag

f = Flag()

grid = f.grid
grid_size = f.grid_size
target_size = f.target_size
threshold = f.threshold


def file_reader(directory, file_name):
    file = open(directory + file_name + '.txt')
    label = file.readlines()
    file.close()
    return label


def label_generator(image_name, file):
    label = []
    for line in file:
        line = line.strip().split()
        name = line[0].split('/')[3]
        if image_name == name:
            label = line[1:]
            break

    """ obj_box: top-left and bottom-right coordinate of the bounding box """
    obj_box = [0, 0, 0, 0]
    for i in range(0, 4):
        obj_box[i] = obj_box[i] + int(float(label[i]) * target_size)

    obj_box = np.array(obj_box)
    return obj_box


def grid_generator(obj_box):
    output = np.zeros(shape=(grid, grid))
    # print(obj_box)
    for i in range(int(obj_box[1] / grid_size), int(obj_box[3] / grid_size) + 1):
        for j in range(int(obj_box[0] / grid_size), int(obj_box[2] / grid_size) + 1):
            x = j * grid_size
            y = i * grid_size
            grid_box = (x, y, x + grid_size, y + grid_size)
            overlap = iou(grid_box=grid_box, obj_box=obj_box)
            if overlap > 0.5:
                output[i, j] = 1

    return output


if __name__ == '__main__':
    image_name = 'ChuangyeguBusstop_Single_Five_color_1.jpg'
    folder_name = find_folder(image_name=image_name)
    image_directory = '../../../../EgoGesture Dataset/' + folder_name + '/'
    label_directory = '../../../../EgoGesture Dataset/label/'

    file = open(label_directory + folder_name + '.txt')
    file1 = file.readlines()
    file.close()

    obj_box = label_generator(image_name=image_name, file=file1)
    output = grid_generator(obj_box=obj_box)
    print(output)

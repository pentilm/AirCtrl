import cv2
import numpy as np
from preprocess.finder import find_folder
from hand_detector.solo.preprocess.solo_flag import Flag
from hand_detector.solo.preprocess.solo_labelgen import label_generator, grid_generator

f = Flag()

grid = f.grid
grid_size = f.grid_size
target_size = f.target_size
threshold = f.threshold


def visualize(image, output):
    """ drawing grid lines """
    for i in range(0, grid + 1):
        image = cv2.line(image, (0, i * grid_size), (grid * grid_size, i * grid_size), (0, 0, 0), 2)
        image = cv2.line(image, (i * grid_size, 0), (i * grid_size, grid * grid_size), (0, 0, 0), 2)

    alpha = .25

    for i in range(0, grid):
        for j in range(0, grid):
            if output[i, j] > threshold:
                glassy_image = image.copy()
                x = j * grid_size
                y = i * grid_size
                glassy_image = cv2.rectangle(glassy_image, (x, y), (x + grid_size, y + grid_size), (240, 20, 200), -1)
                image = cv2.addWeighted(glassy_image, alpha, image, 1 - alpha, 0)

    """ Finding bounding box """
    prediction = np.where(output > threshold)
    row_wise = prediction[0]
    col_wise = prediction[1]
    try:
        x1 = min(col_wise) * grid_size
        y1 = min(row_wise) * grid_size
        x2 = (max(col_wise) + 1) * grid_size
        y2 = (max(row_wise) + 1) * grid_size
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    except ValueError:
        print('Hand is not Found')

    cv2.imshow('solo prediction', image)
    # cv2.imwrite('output_visualize.jpg', image)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    image_name = 'ChuangyeguBusstop_Single_Five_color_526.jpg'
    folder_name = find_folder(image_name)
    image_directory = '../../../EgoGesture Dataset/' + folder_name + '/'
    label_directory = '../../../EgoGesture Dataset/label/'

    image = cv2.imread(image_directory + image_name)
    image = cv2.resize(image, (target_size, target_size))

    file = open(label_directory + folder_name + '.txt')
    label_file = file.readlines()
    file.close()

    obj_box = label_generator(image_name=image_name, file=label_file)
    output = grid_generator(obj_box=obj_box)
    print(output)
    visualize(image=image, output=output)

import os
import cv2
import random
import numpy as np
from preprocess.finder import find_folder
from hand_detector.solo.visualize import visualize
from hand_detector.solo.preprocess.solo_flag import Flag
from hand_detector.solo.preprocess.augmentation import augment, augment_flip
from hand_detector.solo.preprocess.solo_labelgen import file_reader, label_generator, grid_generator

f = Flag()
target_size = f.target_size

dataset_directory = '../../../EgoGesture Dataset/'
label_directory = '../../../EgoGesture Dataset/label/'

folders = ['SingleOne', 'SingleTwo', 'SingleThree', 'SingleFour', 'SingleFive', 'SingleSix', 'SingleSeven',
           'SingleEight']

label = {}
for folder in folders:
    label[folder] = file_reader(label_directory, folder)


def train_generator(steps_per_epoch, sample_per_batch):
    train_image_files = []
    for folder in folders:
        train_image_files = train_image_files + os.listdir(dataset_directory + folder)

    for i in range(0, 15):
        random.shuffle(train_image_files)

    print('Training Dataset Size: ', len(train_image_files))

    while True:
        for i in range(0, steps_per_epoch):
            x_batch = []
            y_batch = []
            start = i * sample_per_batch
            for n in range(start, start + sample_per_batch):
                image_name = train_image_files[n]
                folder_name = find_folder(image_name)
                image = cv2.imread(dataset_directory + folder_name + '/' + image_name)
                image = cv2.resize(image, (target_size, target_size))
                obj_box = label_generator(image_name=image_name, file=label.get(folder_name))

                # 01: original image
                x_batch.append(image)
                output = grid_generator(obj_box=obj_box)
                y_batch.append(output)
                # visualize(image=image, output=output)

                # 02: original image + augment
                image_aug, bbox_aug = augment(image=image, bbox=obj_box)
                x_batch.append(image_aug)
                output = grid_generator(obj_box=bbox_aug)
                y_batch.append(output)
                # visualize(image=image_aug, output=output)

            x_batch = np.asarray(x_batch)
            x_batch = x_batch.astype('float32')
            x_batch = x_batch / 255.0
            y_batch = np.asarray(y_batch)

            yield x_batch, y_batch


if __name__ == '__main__':
    gen = train_generator(steps_per_epoch=10, sample_per_batch=100)
    batch_x, batch_y = next(gen)
    print(batch_x)
    print(batch_y)

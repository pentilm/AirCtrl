import os
import random
import numpy as np
from hand_detector.yolo.utils.utils import visualize
from hand_detector.yolo.preprocess.augmentation import augment, flip
from hand_detector.yolo.preprocess.labelgen import label_generator, bbox_to_grid


def batch_indices(batch_size, dataset_size):
    index_a = list(range(0, dataset_size, batch_size))
    index_b = list(range(batch_size, dataset_size, batch_size))
    index_b.append(dataset_size)
    indices = list(zip(index_a, index_b))
    return indices


def load_train_images():
    folder_names = ['SingleOne', 'SingleTwo', 'SingleThree', 'SingleFour', 'SingleFive', 'SingleSix', 'SingleSeven',
                    'SingleEight']
    train_image_files = []
    dataset_directory = '../../../EgoGesture Dataset/'
    for folder in folder_names:
        train_image_files = train_image_files + os.listdir(dataset_directory + folder + '/')
    return train_image_files


def load_valid_images():
    folder_name = ['SingleOneValid', 'SingleTwoValid', 'SingleThreeValid', 'SingleFourValid', 'SingleFiveValid',
                   'SingleSixValid', 'SingleSevenValid', 'SingleEightValid']
    valid_image_files = []
    dataset_directory = '../../../EgoGesture Dataset/'
    for folder in folder_name:
        valid_image_files = valid_image_files + os.listdir(dataset_directory + folder + '/')
    return valid_image_files


def train_generator(batch_size, is_augment=True):
    if is_augment:
        batch_size = int(batch_size / 4)

    directory = '../../../EgoGesture Dataset/'
    train_image_files = load_train_images()
    dataset_size = len(train_image_files)
    indices = batch_indices(batch_size=batch_size, dataset_size=dataset_size)
    print('Training Dataset Size: ', dataset_size)

    while True:
        for i in range(0, 10):
            random.shuffle(train_image_files)

        for index in indices:
            x_batch = []
            y_batch = []

            for n in range(index[0], index[1]):
                image_name = train_image_files[n]
                image, bbox = label_generator(directory, image_name, type='')
                yolo_out = bbox_to_grid(bbox)
                x_batch.append(image)
                y_batch.append(yolo_out)
                # visualize(image, yolo_out, RGB2BGR=True)

                # augment
                image_aug, bbox_aug = augment(image, bbox)
                yolo_out = bbox_to_grid(bbox_aug)
                x_batch.append(image_aug)
                y_batch.append(yolo_out)
                # visualize(image_aug, yolo_out, RGB2BGR=True)

                # horizontal flip
                image_flip, bbox_flip = flip(image, bbox)
                yolo_out = bbox_to_grid(bbox_flip)
                x_batch.append(image_flip)
                y_batch.append(yolo_out)
                # visualize(image_flip, yolo_out, RGB2BGR=True)

                # horizontal flip + augment
                image_flip_aug, bbox_flip_aug = augment(image_flip, bbox_flip)
                yolo_out = bbox_to_grid(bbox_flip_aug)
                x_batch.append(image_flip_aug)
                y_batch.append(yolo_out)
                # visualize(image_flip_aug, yolo_out, RGB2BGR=True)

            x_batch = np.asarray(x_batch) / 255.0
            y_batch = np.asarray(y_batch)
            yield x_batch, y_batch


def valid_generator(batch_size):
    directory = '../../../EgoGesture Dataset/'
    valid_image_files = load_valid_images()
    dataset_size = len(valid_image_files)
    indices = batch_indices(batch_size=batch_size, dataset_size=dataset_size)
    print('Validation Dataset Size: ', dataset_size)

    while True:
        for i in range(0, 10):
            random.shuffle(valid_image_files)

        for index in indices:
            x_batch = []
            y_batch = []

            for n in range(index[0], index[1]):
                image_name = valid_image_files[n]
                image, bbox = label_generator(directory, image_name, type='Valid')
                yolo_out = bbox_to_grid(bbox)
                x_batch.append(image)
                y_batch.append(yolo_out)
                # visualize(image, yolo_out, RGB2BGR=True)

            x_batch = np.asarray(x_batch) / 255.0
            y_batch = np.asarray(y_batch)
            yield x_batch, y_batch


if __name__ == '__main__':
    gen = train_generator(batch_size=100)
    # gen = valid_generator(batch_size=100)
    batch_x, batch_y = next(gen)
    # print(batch_x)
    # print(batch_y)

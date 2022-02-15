import os
import cv2
import random
import numpy as np
from visualize import visualize
from preprocess.datagen import label_generator
from preprocess.augmentation import rotation, translate, crop, noise, salt


def train_generator(sample_per_batch, batch_number):
    """ Generating training data """
    train_image_file = []
    directory = '../EgoGesture Dataset/'
    folder_name = ['SingleOne', 'SingleTwo', 'SingleThree', 'SingleFour', 'SingleFive',
                   'SingleSix', 'SingleSeven', 'SingleEight']

    for folder in folder_name:
        train_image_file = train_image_file + os.listdir(directory + folder + '/')

    for i in range(0, 10):
        random.shuffle(train_image_file)

    print('Training Dataset Size: {0}'.format(len(train_image_file)))

    while True:
        for i in range(0, batch_number - 1):
            start = i * sample_per_batch
            end = (i + 1) * sample_per_batch
            x_batch = []
            y_batch_prob = []
            y_batch_pos = []
            for n in range(start, end):
                image_name = train_image_file[n]

                try:
                    image, probability, position = label_generator(directory=directory,
                                                                   image_name=image_name,
                                                                   type='')
                except cv2.error:
                    print(image_name)
                    continue

                # 1.0 Original image
                x_batch.append(image)
                y_batch_prob.append(probability)
                pos = np.array([position, ] * 10)
                y_batch_pos.append(pos)
                # visualize(image, probability, pos[0])

                """ Augmentation """

                # 2.0 Original + translate
                im, pos = translate(image, probability, position)
                x_batch.append(im)
                y_batch_prob.append(probability)
                pos = np.array([pos, ] * 10)
                y_batch_pos.append(pos)
                # visualize(im, probability, pos[0])

                # 3.0 Original + rotation
                im, pos = rotation(image, probability, position)
                x_batch.append(im)
                y_batch_prob.append(probability)
                pos = np.array([pos, ] * 10)
                y_batch_pos.append(pos)
                # visualize(im, probability, pos[0])

                # 4.0 Original + salt
                im, pos = salt(image, probability, position)
                x_batch.append(im)
                y_batch_prob.append(probability)
                pos = np.array([pos, ] * 10)
                y_batch_pos.append(pos)
                # visualize(im, probability, pos[0])

                # 5.0 Original + crop
                im, pos = crop(image, probability, position)
                x_batch.append(im)
                y_batch_prob.append(probability)
                pos = np.array([pos, ] * 10)
                y_batch_pos.append(pos)
                # visualize(im, probability, pos)

                # 6.0 Original + noise
                im, pos = noise(image, probability, position)
                x_batch.append(im)
                y_batch_prob.append(probability)
                pos = np.array([pos, ] * 10)
                y_batch_pos.append(pos)
                # visualize(im, probability, pos[0])

                # 7.0 Original + rotate + translate
                im, pos = rotation(image, probability, position)
                im, pos = translate(im, probability, pos)
                x_batch.append(im)
                y_batch_prob.append(probability)
                pos = np.array([pos, ] * 10)
                y_batch_pos.append(pos)
                # visualize(im, probability, pos)

                # 8.0 Original + rotate + crop
                im, pos = rotation(image, probability, position)
                im, pos = crop(im, probability, pos)
                x_batch.append(im)
                y_batch_prob.append(probability)
                pos = np.array([pos, ] * 10)
                y_batch_pos.append(pos)
                # visualize(im, probability, pos[0])

            x_batch = np.asarray(x_batch)
            x_batch = x_batch.astype('float32')
            x_batch = x_batch / 255.

            y_batch_prob = np.asarray(y_batch_prob)
            y_batch_pos = np.asarray(y_batch_pos)
            y_batch_pos = y_batch_pos.astype('float32')
            y_batch_pos = y_batch_pos / 128.
            y_batch = [y_batch_prob, y_batch_pos]
            yield (x_batch, y_batch)


def valid_generator(sample_per_batch, batch_number):
    """ Generating validation data """
    valid_image_file = []
    directory = '../EgoGesture Dataset/'
    folder_name = ['SingleOneValid', 'SingleTwoValid', 'SingleThreeValid', 'SingleFourValid', 'SingleFiveValid',
                   'SingleSixValid', 'SingleSevenValid', 'SingleEightValid']

    for folder in folder_name:
        valid_image_file = valid_image_file + os.listdir(directory + folder + '/')

    # print(len(valid_image_file))

    while True:
        for i in range(0, batch_number - 1):
            start = i * sample_per_batch
            end = (i + 1) * sample_per_batch
            x_batch = []
            y_batch_prob = []
            y_batch_pos = []
            for n in range(start, end):
                image_name = valid_image_file[n]

                try:
                    image, probability, position = label_generator(directory=directory,
                                                                   image_name=image_name,
                                                                   type='Valid')
                except cv2.error:
                    print(image_name)
                    continue

                # 1.0 Original image
                x_batch.append(image)
                y_batch_prob.append(probability)
                pos = np.array([position, ] * 10)
                y_batch_pos.append(pos)

            x_batch = np.asarray(x_batch)
            x_batch = x_batch.astype('float32')
            x_batch = x_batch / 255.

            y_batch_prob = np.asarray(y_batch_prob)
            y_batch_pos = np.asarray(y_batch_pos)
            y_batch_pos = y_batch_pos.astype('float32')
            y_batch_pos = y_batch_pos / 128.
            y_batch = [y_batch_prob, y_batch_pos]
            yield (x_batch, y_batch)


if __name__ == '__main__':
    gen = train_generator(sample_per_batch=100, batch_number=220)
    batch_x, batch_y = next(gen)
    print(batch_x)
    print(batch_y)

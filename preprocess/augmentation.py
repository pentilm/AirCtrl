import cv2
import random
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from preprocess.datagen import label_generator


def rotation(image, prob, keys):
    """ Brightness, rotation, and scaling shear transformation """
    aug = iaa.Sequential()
    aug.add(iaa.Multiply(random.uniform(0.25, 1.5)))
    aug.add(iaa.Affine(rotate=random.uniform(-180, 180),
                       scale=random.uniform(.7, 1.1),
                       shear=random.uniform(-25, 25),
                       cval=(0, 255)))

    seq_det = aug.to_deterministic()

    image_aug = seq_det.augment_images([image])[0]

    keys = ia.KeypointsOnImage([ia.Keypoint(x=keys[0], y=keys[1]),
                                ia.Keypoint(x=keys[2], y=keys[3]),
                                ia.Keypoint(x=keys[4], y=keys[5]),
                                ia.Keypoint(x=keys[6], y=keys[7]),
                                ia.Keypoint(x=keys[8], y=keys[9])], shape=image.shape)

    keys_aug = seq_det.augment_keypoints([keys])[0]
    k = keys_aug.keypoints
    output = [k[0].x, k[0].y, k[1].x, k[1].y, k[2].x, k[2].y, k[3].x, k[3].y, k[4].x, k[4].y]

    index = 0
    for i in range(0, len(prob)):
        output[index] = output[index] * prob[i]
        output[index + 1] = output[index + 1] * prob[i]
        index = index + 2
    output = np.array(output)
    return image_aug, output


def translate(image, prob, keys):
    """ Translating image """
    aug = iaa.Sequential()
    x = random.uniform(-10, 10) * .01
    aug.add(iaa.Affine(translate_percent={"x": x, "y": x},
                       scale=random.uniform(.7, 1.1),
                       rotate=random.uniform(-45, 45),
                       cval=(0, 255)))
    aug.add(iaa.Multiply(random.uniform(0.5, 1.5)))

    seq_det = aug.to_deterministic()

    image_aug = seq_det.augment_images([image])[0]

    keys = ia.KeypointsOnImage([ia.Keypoint(x=keys[0], y=keys[1]),
                                ia.Keypoint(x=keys[2], y=keys[3]),
                                ia.Keypoint(x=keys[4], y=keys[5]),
                                ia.Keypoint(x=keys[6], y=keys[7]),
                                ia.Keypoint(x=keys[8], y=keys[9])], shape=image.shape)

    keys_aug = seq_det.augment_keypoints([keys])[0]
    k = keys_aug.keypoints
    output = [k[0].x, k[0].y, k[1].x, k[1].y, k[2].x, k[2].y, k[3].x, k[3].y, k[4].x, k[4].y]

    index = 0
    for i in range(0, len(prob)):
        output[index] = output[index] * prob[i]
        output[index + 1] = output[index + 1] * prob[i]
        index = index + 2
    output = np.array(output)
    return image_aug, output


def crop(image, prob, keys):
    """ Cropping """
    x = random.randint(0, 5)
    y = random.randint(0, 5)
    r = random.uniform(-5, 5)
    aug = iaa.Sequential([iaa.Crop(px=((0, x), (0, y), (0, x), (0, y))), iaa.Affine(shear=r, cval=(0, 255))])
    aug.add(iaa.Multiply(random.uniform(0.25, 1.5)))
    seq_det = aug.to_deterministic()

    image_aug = seq_det.augment_images([image])[0]

    keys = ia.KeypointsOnImage([ia.Keypoint(x=keys[0], y=keys[1]),
                                ia.Keypoint(x=keys[2], y=keys[3]),
                                ia.Keypoint(x=keys[4], y=keys[5]),
                                ia.Keypoint(x=keys[6], y=keys[7]),
                                ia.Keypoint(x=keys[8], y=keys[9])], shape=image.shape)

    keys_aug = seq_det.augment_keypoints([keys])[0]
    k = keys_aug.keypoints
    output = [k[0].x, k[0].y, k[1].x, k[1].y, k[2].x, k[2].y, k[3].x, k[3].y, k[4].x, k[4].y]

    index = 0
    for i in range(0, len(prob)):
        output[index] = output[index] * prob[i]
        output[index + 1] = output[index + 1] * prob[i]
        index = index + 2
    output = np.array(output)
    return image_aug, output


def noise(image, prob, keys):
    """ Adding noise """
    aug = iaa.Sequential([iaa.Multiply(random.uniform(0.25, 1.5)),
                          iaa.AdditiveGaussianNoise(scale=0.05 * 255)])
    seq_det = aug.to_deterministic()

    image_aug = seq_det.augment_images([image])[0]

    keys = ia.KeypointsOnImage([ia.Keypoint(x=keys[0], y=keys[1]),
                                ia.Keypoint(x=keys[2], y=keys[3]),
                                ia.Keypoint(x=keys[4], y=keys[5]),
                                ia.Keypoint(x=keys[6], y=keys[7]),
                                ia.Keypoint(x=keys[8], y=keys[9])], shape=image.shape)

    keys_aug = seq_det.augment_keypoints([keys])[0]
    k = keys_aug.keypoints
    output = [k[0].x, k[0].y, k[1].x, k[1].y, k[2].x, k[2].y, k[3].x, k[3].y, k[4].x, k[4].y]

    index = 0
    for i in range(0, len(prob)):
        output[index] = output[index] * prob[i]
        output[index + 1] = output[index + 1] * prob[i]
        index = index + 2
    output = np.array(output)
    return image_aug, output


def salt(image, prob, keys):
    """ Adding salt noise """
    r = random.uniform(1, 5) * 0.05
    aug = iaa.Sequential([iaa.Dropout(p=(0, r)), iaa.CoarseDropout(p=0.001, size_percent=0.01),
                          iaa.Salt(0.001), iaa.AdditiveGaussianNoise(scale=0.1 * 255)])
    aug.add(iaa.Multiply(random.uniform(0.25, 1.5)))
    x = random.randrange(-10, 10) * .01
    y = random.randrange(-10, 10) * .01
    aug.add(iaa.Affine(scale=random.uniform(.7, 1.1), translate_percent={"x": x, "y": y}, cval=(0, 255)))

    seq_det = aug.to_deterministic()

    image_aug = seq_det.augment_images([image])[0]

    keys = ia.KeypointsOnImage([ia.Keypoint(x=keys[0], y=keys[1]),
                                ia.Keypoint(x=keys[2], y=keys[3]),
                                ia.Keypoint(x=keys[4], y=keys[5]),
                                ia.Keypoint(x=keys[6], y=keys[7]),
                                ia.Keypoint(x=keys[8], y=keys[9])], shape=image.shape)

    keys_aug = seq_det.augment_keypoints([keys])[0]
    k = keys_aug.keypoints
    output = [k[0].x, k[0].y, k[1].x, k[1].y, k[2].x, k[2].y, k[3].x, k[3].y, k[4].x, k[4].y]

    index = 0
    for i in range(0, len(prob)):
        output[index] = output[index] * prob[i]
        output[index + 1] = output[index + 1] * prob[i]
        index = index + 2
    output = np.array(output)
    return image_aug, output


if __name__ == '__main__':
    directory = '../../EgoGesture Dataset/'
    image_name = 'ChuangyeguBusstop_Single_Five_color_1.jpg'
    image, probability, position = label_generator(directory=directory, image_name=image_name, type='')

    """ augmentation """
    image, position = rotation(image, probability, position)
    # image, position = translate(image, probability, position)
    # image, position = crop(image, probability, position)
    # image, position = flip_horizontal(image, probability, position)
    # image, position = flip_vertical(image, probability, position)
    # image, position = noise(image, probability, position)
    # image, position = salt(image, probability, position)
    print(probability)
    print(position)

    # draw augmented keypoints
    index = 0
    color = [(120, 20, 240), (240, 55, 210), (240, 55, 140), (240, 75, 55), (170, 240, 55)]
    for c, p in enumerate(probability):
        if p > 0.5:
            image = cv2.circle(image, (int(position[index]), int(position[index + 1])), radius=5,
                               color=color[c], thickness=-2)
        index = index + 2
    cv2.imshow('', image)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

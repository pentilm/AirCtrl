import cv2
import random
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from hand_detector.yolo.utils.utils import visualize
from hand_detector.yolo.preprocess.yolo_flag import Flag
from hand_detector.yolo.preprocess.labelgen import label_generator

f = Flag()
size = f.target_size


def augment(image, bbox):
    x = random.randint(-60, 60)
    y = random.randint(-60, 60)
    aug = iaa.Sequential([iaa.AdditiveGaussianNoise(scale=random.uniform(.001, .01) * 255),  # gaussian noise
                          iaa.Multiply(random.uniform(0.5, 1.5)),  # brightness
                          iaa.Affine(translate_px={"x": x, "y": y},  # translation
                                     scale=random.uniform(0.5, 1.5),  # zoom in and out
                                     rotate=random.uniform(-25, 25),  # rotation
                                     shear=random.uniform(-5, 5),  # shear transformation
                                     cval=(0, 255))])  # fill the empty space with color

    aug.add(iaa.Salt(.001))
    bbs = ia.BoundingBoxesOnImage([ia.BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])], shape=image.shape)
    aug = aug.to_deterministic()
    image_aug = aug.augment_image(image)
    bbs_aug = aug.augment_bounding_boxes([bbs])[0]
    b = bbs_aug.bounding_boxes
    bbs_aug = [b[0].x1, b[0].y1, b[0].x2, b[0].y2]
    bbs_aug = np.asarray(bbs_aug)

    bbs_aug[0] = bbs_aug[0] if bbs_aug[0] > 0 else 0
    bbs_aug[1] = bbs_aug[1] if bbs_aug[1] > 0 else 0
    bbs_aug[2] = bbs_aug[2] if bbs_aug[2] < size else size
    bbs_aug[3] = bbs_aug[3] if bbs_aug[3] < size else size
    return image_aug, bbs_aug


def flip(image, bbox):
    aug = iaa.Sequential([iaa.Fliplr(1.0)])

    bbs = ia.BoundingBoxesOnImage([
        ia.BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])], shape=image.shape)

    aug = aug.to_deterministic()
    image_aug = aug.augment_image(image)
    image_aug = image_aug.copy()
    bbs_aug = aug.augment_bounding_boxes([bbs])[0]
    b = bbs_aug.bounding_boxes
    bbs_aug = [b[0].x1, b[0].y1, b[0].x2, b[0].y2]
    bbs_aug = np.asarray(bbs_aug)

    bbs_aug[0] = bbs_aug[0] if bbs_aug[0] > 0 else 0
    bbs_aug[1] = bbs_aug[1] if bbs_aug[1] > 0 else 0
    bbs_aug[2] = bbs_aug[2] if bbs_aug[2] < size else size
    bbs_aug[3] = bbs_aug[3] if bbs_aug[3] < size else size
    return image_aug, bbs_aug


if __name__ == '__main__':
    dir = '../../../../EgoGesture Dataset/'
    img_name = 'BasketballField_Single_Eight_color_91.jpg'
    img, box = label_generator(dir, img_name, type='')

    # image_aug, bbox_aug = img, box
    img_aug, bbox_aug = augment(image=img, bbox=box)
    # image_aug, bbox_aug = flip(image=img, bbox=box)
    # image_aug, bbox_aug = augment(image=image_aug, bbox=bbox_aug)
    bbox_aug = [int(b) for b in bbox_aug]

    x1, y1, x2, y2 = bbox_aug[0], bbox_aug[1], bbox_aug[2], bbox_aug[3]
    img_aug = cv2.rectangle(img_aug, (x1, y1), (x2, y2), f.box_color, 3)
    visualize(image=img_aug, title='visualize', RGB2BGR=True)

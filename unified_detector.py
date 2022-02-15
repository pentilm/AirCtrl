import cv2
import numpy as np
from net.network import model


class Fingertips:
    def __init__(self, weights):
        self.model = model()
        self.model.load_weights(weights)

    @staticmethod
    def class_finder(prob):
        cls = ''
        classes = [0, 1, 2, 3, 4, 5, 6, 7]

        if np.array_equal(prob, np.array([0, 1, 0, 0, 0])):
            cls = classes[0]
        elif np.array_equal(prob, np.array([0, 1, 1, 0, 0])):
            cls = classes[1]
        elif np.array_equal(prob, np.array([0, 1, 1, 1, 0])):
            cls = classes[2]
        elif np.array_equal(prob, np.array([0, 1, 1, 1, 1])):
            cls = classes[3]
        elif np.array_equal(prob, np.array([1, 1, 1, 1, 1])):
            cls = classes[4]
        elif np.array_equal(prob, np.array([1, 0, 0, 0, 1])):
            cls = classes[5]
        elif np.array_equal(prob, np.array([1, 1, 0, 0, 1])):
            cls = classes[6]
        elif np.array_equal(prob, np.array([1, 1, 0, 0, 0])):
            cls = classes[7]
        return cls

    def classify(self, image):
        image = np.asarray(image)
        image = cv2.resize(image, (128, 128))
        image = image.astype('float32')
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        probability, position = self.model.predict(image)
        probability = probability[0]
        position = position[0]
        return probability, position

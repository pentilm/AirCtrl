from keras.models import Model
from keras.applications import VGG16
from keras.layers import Conv2D, Reshape


def model():
    model = VGG16(include_top=False, input_shape=(416, 416, 3))
    x = model.output
    x = Conv2D(1, (1, 1), activation='sigmoid')(x)
    output = Reshape((13, 13), name='output')(x)
    model = Model(model.input, output)
    return model


if __name__ == '__main__':
    model = model()
    model.summary()

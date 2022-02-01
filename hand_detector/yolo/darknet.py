from keras.models import Input, Model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation


def conv_batch_norm_relu(x, n_filters, f, padding='same', activation='relu'):
    x = Conv2D(n_filters, f, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x


def model():
    input = Input(shape=(224, 224, 3))
    x = conv_batch_norm_relu(input, 32, (3, 3), padding='same', activation='relu')
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = conv_batch_norm_relu(x, 64, (3, 3), padding='same', activation='relu')
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = conv_batch_norm_relu(x, 128, (3, 3), padding='same', activation='relu')
    x = conv_batch_norm_relu(x, 64, (1, 1), padding='same', activation='relu')
    x = conv_batch_norm_relu(x, 128, (3, 3), padding='same', activation='relu')
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = conv_batch_norm_relu(x, 256, (3, 3), padding='same', activation='relu')
    x = conv_batch_norm_relu(x, 128, (1, 1), padding='same', activation='relu')
    x = conv_batch_norm_relu(x, 256, (3, 3), padding='same', activation='relu')
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = conv_batch_norm_relu(x, 512, (3, 3), padding='same', activation='relu')
    x = conv_batch_norm_relu(x, 256, (1, 1), padding='same', activation='relu')
    x = conv_batch_norm_relu(x, 512, (3, 3), padding='same', activation='relu')
    x = conv_batch_norm_relu(x, 256, (1, 1), padding='same', activation='relu')
    x = conv_batch_norm_relu(x, 512, (3, 3), padding='same', activation='relu')
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = conv_batch_norm_relu(x, 1024, (3, 3), padding='same', activation='relu')
    x = conv_batch_norm_relu(x, 512, (1, 1), padding='same', activation='relu')
    x = conv_batch_norm_relu(x, 1024, (3, 3), padding='same', activation='relu')
    x = conv_batch_norm_relu(x, 512, (1, 1), padding='same', activation='relu')
    x = conv_batch_norm_relu(x, 1024, (3, 3), padding='same', activation='relu')
    x = Conv2D(5, (1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid', name='output')(x)
    return Model(inputs=input, outputs=x)


if __name__ == '__main__':
    model = model()
    model.summary()

import tensorflow as tf
from net.network import model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from generator import train_generator, valid_generator


def loss_function_1(y_true, y_pred):
    """ Probabilistic output loss """
    a = tf.clip_by_value(y_pred, 1e-20, 1)
    b = tf.clip_by_value(tf.subtract(1.0, y_pred), 1e-20, 1)
    cross_entropy = - tf.multiply(y_true, tf.log(a)) - tf.multiply(tf.subtract(1.0, y_true), tf.log(b))
    cross_entropy = tf.reduce_mean(cross_entropy, 0)
    loss = tf.reduce_mean(cross_entropy)
    return loss


def loss_function_2(y_true, y_pred):
    """ Positional output loss """
    square_diff = tf.squared_difference(y_true, y_pred)
    mask = tf.not_equal(y_true, 0)
    mask = tf.cast(mask, tf.float32)
    square_diff = tf.multiply(square_diff, mask)
    square_diff = tf.reduce_mean(square_diff, 1)
    square_diff = tf.reduce_mean(square_diff, 0)
    loss = tf.reduce_mean(square_diff)
    return loss


# Creating the model
model = model()
model.summary()

# Compile
adam = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-10, decay=0.0)
loss_function = {"probabilistic_output": loss_function_1, "positional_output": loss_function_2}
metrics = {"probabilistic_output": loss_function_1, "positional_output": loss_function_2}
model.compile(optimizer=adam, loss=loss_function, metrics=metrics)

# Train
epochs = 100
train_gen = train_generator(sample_per_batch=8, batch_number=3136)
val_gen = valid_generator(sample_per_batch=64, batch_number=20)

checkpoints = ModelCheckpoint('weights/performance{epoch:03d}.h5', save_weights_only=True, period=1)
history = model.fit_generator(train_gen, steps_per_epoch=3136, epochs=epochs, verbose=1,
                              validation_data=val_gen, validation_steps=20,
                              shuffle=True, callbacks=[checkpoints], max_queue_size=100)

with open('history.txt', 'a+') as f:
    print(history.history, file=f)

print('All Done!')

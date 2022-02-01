import os
from math import floor
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from hand_detector.solo.solo_net import model
from hand_detector.solo.generator import train_generator

# create model
model = model()
model.summary()

# compile
adam = Adam(lr=1e-5)
model.compile(optimizer=adam, loss='binary_crossentropy')

# train
epochs = 10
batch_size = 4
train_set_size = 36071
steps_per_epoch = int(floor(train_set_size / batch_size))
train_gen = train_generator(steps_per_epoch=steps_per_epoch, sample_per_batch=batch_size)

weights_directory = '../../weights/'
if not os.path.exists(weights_directory):
    os.makedirs(weights_directory)

checkpoints = ModelCheckpoint(weights_directory + 'hand_weights{epoch:03d}.h5', save_weights_only=True, period=1)
history = model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1,
                              shuffle=True, callbacks=[checkpoints], max_queue_size=32)

with open('history.txt', 'a+') as f:
    print(history.history, file=f)

print('All Done!')


#%%
import tensorflow as tf
import numpy as np
import networks
import datetime
import os
import cv2
import importlib
import training_utils
import data_utils
#%%
#__(1)__________Load Data_______________
TRAIN_DIR = './HELEN/processed_samples/train'
TEST_DIR = './HELEN/processed_samples/test'

size_args = (256, 64, 2.5)

training_generator = data_utils.LandmarkImageGenerator(TRAIN_DIR, *size_args)
testing_generator = data_utils.LandmarkImageGenerator(TEST_DIR, *size_args)

train_dataset = training_utils.FLDR_preprocessed_datagen(training_generator)
test_dataset = training_utils.FLDR_preprocessed_datagen(testing_generator)


#%%
#__(2)__________Load Model_______________
NUM_FEATURES = 194
IMG_SHAPE = (256,256,3)

rccnn = networks.RCCNN(NUM_FEATURES, IMG_SHAPE)

#__(5)_________Optimizer__________________

optim = tf.keras.optimizers.Adam(1e-5)

rccnn.compile(optimizer = optim, loss = RCCNN_loss, metrics = 'val_loss')

#__(7)__________Set up callbacks___________
CHECKPOINT_DIR = './checkpoints'
checkpoint_prefix  = os.path.join(CHECKPOINT_DIR, 'ckpt_{epoch}')


CALLBACKS = [
    tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            monitor = 'val_loss',
            save_weights_only=True,
            save_best_only = True,
            mode = 'min'),
    tf.keras.callbacks.TensorBoard(
        './logs',
        histogram_freq = 1),
    training_utils.FLDRegressionCallback('./examples/', testing_generator)
    tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience = 5, min_lr = 1e-6),
]

rccnn.fit(train_dataset, 
        epochs = 100, 
        steps_per_epoch = 2000,
        validation_data = test_dataset,
        validation_steps = 300,
        callbacks = CALLBACKS,
        use_multiprocessing = True)









import tensorflow as tf
import numpy as np
import data_utils
import networks
from training_utils import RCCNN_loss, FLDRegressionCallback
import datetime
import os
import cv2

#__(1)__________Load Data_______________
TRAINING_DIR = './HELEN/processed_samples/train/'
TESTING_DIR = './HELEN/processed_samples/test/'

trainX, trainY = data_utils.load_dataset(TRAINING_DIR)
testX, testY = data_utils.load_dataset(TESTING_DIR)

BATCH_SIZE = 64

train_dataset = tf.data.Dataset.from_tensor_slices((trainX, trainY))
train_dataset.shuffle(buffer_size = 400).batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices((testX, testY))
test_dataset.batch(BATCH_SIZE)

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
    FLDRegressionCallback('./examples','./HELEN/processed_samples/test/'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience = 5, min_lr = 1e-6),
]

rccnn.fit(train_dataset, 
        epochs = 100, 
        validation_data = test_dataset,
        callbacks = CALLBACKS)








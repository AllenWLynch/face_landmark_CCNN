
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

#BAD GENERATOR STUFF
TRAIN_DIR = './HELEN/HELEN_dataset/train'
TEST_DIR = './HELEN/HELEN_dataset/test'


'''train_generator = training_utils.KeypointsDataset(TRAIN_DIR)
train_dataset = training_utils.TFGenerator(train_generator, 
                    (tf.float32, (tf.float32, tf.float32)),
                    (tf.TensorShape([256,256,3]), tf.TensorShape([64,64,194]), tf.TensorShape([194,2])), 
                    training_utils.FLL_preprocces(3), 
                    8, 5)


#test_generator = training_utils.KeypointsDataset(TEST_DIR)
#test_dataset = training_utils.TFGenerator(test_generator, (tf.float32, (tf.float32, tf.float32)),
#                    (tf.TensorShape([256,256,3]), (tf.TensorShape([64,64,194]),tf.TensorShape([194,2]))), 
#                    training_utils.FLL_preprocces(3), 
#                    8, 5)'''

#__(1a)____ Load data___________________

X,Y = training_utils.load_HELEN_dataset(TRAIN_DIR, 3, 0.25)

#%%
#__(2)__________Load Model_______________
NUM_FEATURES = 194
IMG_SHAPE = (256,256,3)

rccnn = networks.RCCNN(NUM_FEATURES, IMG_SHAPE)

#__(5)_________Optimizer__________________

optim = tf.keras.optimizers.Adam(1e-5)

rccnn.compile(optimizer = optim, 
            loss = {
                'heatmap_output' : tf.keras.losses.CategoricalCrossentropy(), 
                'regression_output' : tf.keras.losses.MeanSquaredError()},
            metrics = {
                'heatmap_output' : [tf.keras.metrics.CategoricalCrossentropy()],
                'regression_output' : [tf.keras.metrics.MeanSquaredError()]
            })

#__(7)__________Set up callbacks___________
CHECKPOINT_DIR = './checkpoints'
checkpoint_prefix  = os.path.join(CHECKPOINT_DIR, 'ckpt_{epoch}')


CALLBACKS = [
    tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True),
    tf.keras.callbacks.TensorBoard(
        'logs',
        update_freq = 50),
    #training_utils.FLDRegressionCallback('./examples/', test_generator),
    tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience = 5, min_lr = 1e-6),]

rccnn.fit(
            x = X,
            y = Y,
            batch_size = 12,
            epochs = 50,
            callbacks = CALLBACKS,
            validation_split = 0.05,
)











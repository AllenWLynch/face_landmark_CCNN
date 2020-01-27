
#%%
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import numpy as np
import networks
import cv2
import training_utils
import importlib
import os
import datetime

#%%
training_utils = importlib.reload(training_utils)
#%%


#%%
if __name__ == "__main__":

#__(0)___ Define policy
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)

#__(1)___ Load in data


    TRAIN_DIR = './HELEN/HELEN_dataset/train'
    TEST_DIR = './HELEN/HELEN_dataset/test'

    dataset_args = ((tf.float32, (tf.float32, tf.float32)),
                        (tf.TensorShape([256,256,3]), (tf.TensorShape([64,64,194]), tf.TensorShape([194,2]))), 
                        training_utils.FLL_preprocces(3), 
                        16, 5)

    train_dataset = training_utils.TFGenerator(training_utils.KeypointsDataset(TRAIN_DIR), *dataset_args)
                        
    test_dataset = training_utils.TFGenerator(training_utils.KeypointsDataset(TEST_DIR), *dataset_args)
    
#%%
    #__(2)___ Define model

    NUM_FEATURES = 194
    IMG_SHAPE = (256,256,3)

    rccnn = networks.RCCNN(NUM_FEATURES, IMG_SHAPE)

#%%
#__(3)___ Define model checkpointer

    LOAD_FROM_CHECKPOINT = False
    CHECKPOINT_DIR = './checkpoints'

    checkpoint = tf.train.Checkpoint(landmark_cascade = rccnn)

    if LOAD_FROM_CHECKPOINT:
        try:
            checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR)).assert_consumed()
        except Exception as err:
            print('Failed to load models from checkpoint!')
            raise err

    manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_DIR, max_to_keep=5)

#__(4)____ Set up tensorboard

    LOGDIR = 'logs'

    log_writer = tf.summary.create_file_writer(
            os.path.join(
                LOGDIR,
                "fit/",
                datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            ))
#%%
#__(5)____ Train it

    EPOCHS = 100
    STEPS_PER_EPOCH = 200
    EVAL_STEPS = 10
    LOGSTEPS = 10
    CHECKPOINT_EVERY = 3

    heatmap_loss_obj = tf.keras.losses.CategoricalCrossentropy()
    regression_loss_obj = tf.keras.losses.MeanSquaredError()

    heatmap_metrics = [tf.keras.metrics.CategoricalCrossentropy()]
    regression_metrics = [tf.keras.metrics.MeanSquaredError()]

    LEARNING_RATE = 1e-5
    optim = tf.keras.optimizers.Adam(LEARNING_RATE)
    optim = mixed_precision.LossScaleOptimizer(optim, loss_scale = 'dynamic')

    trainer = training_utils.FLLTrainer(rccnn, optim, heatmap_loss_obj, regression_loss_obj, log_writer, 
            heatmap_metrics, regression_metrics)

#%%
    trainer.fit(train_dataset, test_dataset, EPOCHS, STEPS_PER_EPOCH, EVAL_STEPS, manager, CHECKPOINT_EVERY)

#%%
import tensorflow as tf
import numpy as np
import cv2
from data_utils import show_keypoints, prepare_input, LandmarkImageGenerator
import os
#%%
#sum of crossentropy, normalized over batch dimension
#sum of crossentropy, normalized over batch dimension
def heatmap_loss(y_true, y_pred):
    (_, d, h, w, n) = y_pred.get_shape()
    log_p_hat = tf.math.log(y_pred)
    return -1. * tf.reduce_mean(tf.multiply(y_true, log_p_hat))

class FLDRegressionCallback(tf.keras.callbacks.Callback):

    def __init__(self, save_dir, test_datagen):
        newfilename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.save_dir = os.path.join(save_dir, newfilename)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        assert(os.path.exists(image_dir))
        self.test_datagen = test_datagen
    
    def on_epoch_end(self, epoch, logs = None):
        
        img, keypoints = next(iter(self.test_datagen))

        heatmap, regression = self.model(np.array([img]))

        regression = regression[0][-1]

        coordinates = (regression + 1) * 128. 

        coordinates = np.clip(coordinates, 0, 256)

        img = show_keypoints(unnorm_img, coordinates)

        cv2.imwrite(os.path.join(self.save_dir, 'epoch_{}.jpg'.format(str(epoch))), img)

#%%
def FLDR_preprocessed_datagen(image_generator, batch_size = 64):

    dataset = tf.data.Dataset.from_generator(
        image_generator,
        (tf.float32, (tf.float32, tf.float32)),
        (tf.TensorShape([256,256,3]), (tf.TensorShape([3,64,64,194]), tf.TensorShape([3,194,2])))
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(4)

    return dataset
    

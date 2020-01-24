#%%
import tensorflow as tf
import numpy as np
import cv2
from data_utils import show_keypoints, prepare_input, LandmarkImageGenerator
import os
#%%
#sum of crossentropy, normalized over batch dimension
#sum of crossentropy, normalized over batch dimension
def heatmap_loss(H_hat, H):

    #crossentropy loss, expection over features axis
    #shapes = H_hat = (m, d, h, w, N), H = (m, h, w, N)
    (m,d,h,w,N) = H_hat.get_shape()
    H = tf.expand_dims(H, 1)
    log_p_hat = tf.math.log(H_hat)
    cross_entropy = - tf.multiply(H, H_hat)
    loss = tf.stop_gradient(1/(m*d*N*h*w)) * tf.reduce_sum(cross_entropy)
    return loss

def regression_loss(R_hat, R):
    (m, d, k, c) = R_hat.get_shape()
    R = tf.expand_dims(R, 1)
    return tf.stop_gradient(1/(m*d*c*k)) * tf.reduce_sum(tf.square(R_hat - R))


def RCCNN_loss(prediction, target):
    
    H_hat, R_hat = prediction
    H_real, R_real = target

    return 0.5 * regression_loss(R_hat, R_real) + 0.5 * heatmap_loss(H_hat, H_real)


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
    )

    dataset.batch(batch_size)
    dataset.prefetch(4)

    return dataset
    

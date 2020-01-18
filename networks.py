
import tensorflow as tf
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import cv2
import os


#this sets up the Gaussian distribution for the heatmap
class heat_map_targeter():

    def __init__(self, heatmap_size, std):
        self.heatmap_size = heatmap_size
        self.std = std
        self.distribution = multivariate_normal(cov=[[std**2, 0],[0, std**2]])

    def __call__(self, targets):
        num_points, coors = targets.shape
        assert(coors == 2), 'Array must be of size (num_points, 2)'
        X_targs, Y_targs = targets[:, 0].reshape(-1,1,1), targets[:,1].reshape(-1,1,1)

        x_mesh, y_mesh = np.meshgrid(np.arange(self.heatmap_size), np.arange(self.heatmap_size))
        x_error = np.expand_dims(np.expand_dims(x_mesh, 0) - X_targs, -1)
        y_error = np.expand_dims(np.expand_dims(y_mesh, 0) - Y_targs, -1)

        errors = np.concatenate((x_error, y_error), axis = -1)

        probs = self.distribution.pdf(errors)
        
        probs = np.transpose(probs, axes = [1,2,0])

        return probs

#sum of crossentropy, normalized over batch dimension
def heatmap_loss(H_hat, H):
    #crossentropy loss, expection over features axis
    #shapes = (m, h, w, N)
    (m,h,w,N) = H_hat.get_shape()
    log_p_hat = tf.math.log(H_hat)
    cross_entropy = - tf.multiply(H, H_hat)
    loss = tf.stop_gradient(1/m) * tf.reduce_sum(cross_entropy)

    return loss

#REGRESSION LOSS, is just MSE between guess and adjusted annotations

#processing pipeline for a sample
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def input_pipeline(annotation_file, path_to_images):
    with open(annotation_file) as f:
        annotations = f.readlines()
    jpg_filename = annotations[0]
    keypoints = np.array([
        [float(coor) for coor in line.split(' , ')] for line in annotations[1:]
    ])
    img = cv2.imread(os.path.join(path_to_images, jpg_filename))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    



print(input_pipeline('./HELEN/annotation/1.txt'))


    


    

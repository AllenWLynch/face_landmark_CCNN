#%%
import tensorflow as tf
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import cv2
import os
import re
from time import sleep
from scipy import ndimage
from PIL import Image
import random


#%%

#this sets up the Gaussian distribution for the heatmap
class heatmap_targeter():

    def __init__(self, heatmap_size, std, target_size = 256):
        self.heatmap_size = heatmap_size
        self.std = std
        self.target_size = target_size
        self.distribution = multivariate_normal(cov=[[std**2, 0],[0, std**2]])

    def __call__(self, targets):
        targets = targets * self.heatmap_size/self.target_size
        num_points, coors = targets.shape
        assert(coors == 2), 'Array must be of size (num_points, 2)'
        X_targs, Y_targs = targets[:, 0].reshape(-1,1,1), targets[:,1].reshape(-1,1,1)

        x_mesh, y_mesh = np.meshgrid(np.arange(self.heatmap_size), np.arange(self.heatmap_size))
        x_error = np.expand_dims(np.expand_dims(x_mesh, 0) - X_targs, -1)
        y_error = np.expand_dims(np.expand_dims(y_mesh, 0) - Y_targs, -1)

        errors = np.concatenate((x_error, y_error), axis = -1)

        probs = self.distribution.pdf(errors)

        per_feature_sums = np.sum(probs, axis = (-2,-1))

        probs = probs / per_feature_sums[:,np.newaxis, np.newaxis]
        
        probs = np.transpose(probs, axes = [1,2,0])

        return probs

#%%
def show_keypoints(img, keypoints, color = [0.,255.,0.]):
    kp_img = np.copy(img)
    int_keypoints = keypoints.astype(int).T
    kp_img[int_keypoints[1],int_keypoints[0]] = color
    return kp_img
# %%

def load_dataset(directory):

    assert(os.path.exists(directory))
    
    X = []
    Y = []
    for filename in os.listdir(os.path.join(directory, 'keypoints')):

        withoutextension = filename[:-4]

        gaussian = np.load(os.path.join(directory, 'gaussians/',withoutextension + '.npy')).astype('float32')
        keypoints = np.load(os.path.join(directory, 'keypoints/', withoutextension + '.npy')).astype('float32')
        image = cv2.imread(os.path.join(directory, 'images/', withoutextension + '.jpg')).astype('float32')

        norm_image = (image / 255.) - 1.

        norm_keypoints = (keypoints / 128.) - 1.

        X.append(norm_image)
        Y.append((gaussian, norm_keypoints))

    return X, Y


#%%
#Make input pipeline
def rotate_img(rotation_max, true_width, true_height, img, keypoints):

    angle = rotation_max*2*np.random.rand() - rotation_max

    center = np.array([[true_width/2., true_height/2.]])

    theta = angle * 2 * np.pi / 360.
    rotation_mat = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    img = ndimage.rotate(img, -angle, reshape = False)

    keypoints = np.dot(keypoints - center, rotation_mat) + center

    return img, keypoints

def stochastic_padding(keypoints, height, width, face_area_ratio, xmin, ymin, xmax, ymax, true_width, true_height):
    assert(xmin < xmax)
    assert(ymin < ymax)

    smin = max(height, width)
    smax = int(np.sqrt(height * width /face_area_ratio)) #largets s possible

    x_limit = min(true_width-1, xmin + smax) - max(0, xmax - smax)
    y_limit = min(true_height-1, ymin + smax) - max(0, ymax - smax)

    smax = min(x_limit, y_limit, smax)

    if smax <= smin:
        square_sidelength = smax
    else:
        square_sidelength = np.random.randint(int(smin), int(smax))

    if xmax - square_sidelength >= xmin:
        square_x = xmin
    else:
        square_x = np.random.randint(xmax - square_sidelength, xmin)

    if ymax - square_sidelength >= ymin:
        square_y = ymax - square_sidelength
    else:
        square_y = np.random.randint(ymax - square_sidelength, ymin)

    keypoints = keypoints - np.array([[square_x, square_y]])

    return keypoints, (square_x, square_y, square_sidelength)

def rescale_image_keypoints(max_dimension, img, keypoints):
    pass

def prepare_input(annotation_filename, image_dir):

    FOREHEAD_SCALE = 1.4
    ROTATION_MAX = 10
    FACE_FRAME_RATIO = 0.55

    with open(annotation_filename) as f:
        corresponding_jpg = f.readline().strip()

    keypoints = np.loadtxt(annotation_filename, dtype = np.float32, delimiter=',',skiprows=1)
    
    if keypoints.shape != (194,2):
        raise AssertionError('Keypoints file contains wrong number of points!: {}'.format(annotation_filename))
    
    image_path = os.path.join(image_dir, corresponding_jpg + '.jpg')
    assert(os.path.exists(image_path))

    img = cv2.imread(image_path)
    true_height, true_width = img.shape[:2]

    img, keypoints = rotate_img(ROTATION_MAX, true_width, true_height, img, keypoints)

    ((xmin, ymin), (xmax, ymax)) = np.amin(keypoints.astype(int), 0), np.amax(keypoints.astype(int) + 1, 0)

    ymin = max(0, int(ymax - FOREHEAD_SCALE * (ymax - ymin)))
    height = ymax - ymin
    width = xmax - xmin
    
    keypoints, (sx, sy, length) = stochastic_padding(keypoints, height, width, FACE_FRAME_RATIO, 
                xmin, ymin, xmax, ymax, true_width, true_height)

    cropped = img[sy:sy+length, sx:sx+length]

    #just augment the cropped images now

    return cropped, keypoints

TEST_ANNOTATION = './HELEN/annotation/2.txt'
IMAGE_DIR = './HELEN/train'
ANNOTATION_DIR = './HELEN/annotation/'

'''
image, pts, upper, lower = prepare_input(TEST_ANNOTATION, IMAGE_DIR)

with_pts = show_keypoints(image, pts)
with_pts_and_rect = cv2.rectangle(with_pts, upper, lower, (255,0,0), 2)

cv2.imshow('mat', with_pts_and_rect)
cv2.waitKey()'''
files = os.listdir(ANNOTATION_DIR)
for i in range(50):
    try:
        random_seed = random.randint(0,1000)
        np.random.seed(136)
        #r = np.random.randint(0, len(files))
        #filepath = os.path.join(ANNOTATION_DIR, files[r])
        img, pts = prepare_input('./HELEN/annotation/768.txt', IMAGE_DIR)
        withpts = show_keypoints(img, pts)
        cv2.imshow('hey', withpts)
        cv2.waitKey()
    except Exception as err:
        print(filepath, random_seed)
        raise err

# %%

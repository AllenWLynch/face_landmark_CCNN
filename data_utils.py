#%%
import numpy as np
from scipy.stats import multivariate_normal
import cv2
import os
from scipy import ndimage
import random
import matplotlib.pyplot as plt


#%%

class HeatmapTargeter():

    def __init__(self, image_size, heatmap_size, std):

        self.heatmap_size = heatmap_size
        self.target_size = image_size
        self.std = std 
        self.distribution = multivariate_normal(cov=[[std**2, 0],[0, std**2]])

        xx, yy = np.meshgrid(np.arange(2*self.heatmap_size), np.arange(2*self.heatmap_size))
        xx -= self.heatmap_size
        yy -= self.heatmap_size

        errors = np.concatenate([np.expand_dims(xx, -1), np.expand_dims(yy, -1)], axis = -1)
        
        self.probs = self.distribution.pdf(errors)
        self.midpoint = np.array([self.heatmap_size, self.heatmap_size])

    def __call__(self, keypoints):
        #keypoints (N, 2)
        #probs (256,256)
        keypoints = keypoints * (self.heatmap_size/self.target_size)
        N,_ = keypoints.shape
        shifted_gaussians = np.empty((self.heatmap_size, self.heatmap_size, N))
        for (i,keypoint) in enumerate(np.clip(keypoints.astype(int), 0, self.heatmap_size - 1)):
            (cy, cx) = (self.midpoint - keypoint)
            slice = self.probs[cx : cx + self.heatmap_size, cy : cy + self.heatmap_size]
            shifted_gaussians[:,:,i] = slice/np.sum(slice)
        
        return shifted_gaussians

#%%
def show_keypoints(img, keypoints, color = [0.,255.,0.]):
    kp_img = np.copy(img)
    int_keypoints = np.clip(keypoints.astype(int).T, 0, 256-1)
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

def calculate_max_windowsize(max_dim, window_max, xmin, xmax):
    return min(xmin + window_max, max_dim - xmin) + (xmax - max(xmax - window_max, 0)) - (xmax - xmin)

def corner_bounds(max_dim, window_size, xmin, xmax):
    rightmost = min(max_dim - window_size, xmin)
    leftmost = max(0, xmax - window_size)
    return (leftmost, rightmost)

def stochastic_padding(keypoints, height, width, face_area_ratio, xmin, ymin, xmax, ymax, true_width, true_height):
    assert(xmin < xmax)
    assert(ymin < ymax)

    smin = max(height, width)
    smax = int(np.sqrt(height * width /face_area_ratio)) #largets s possible

    x_limit = calculate_max_windowsize(true_width - 1, smax, xmin, xmax)
    y_limit = calculate_max_windowsize(true_height - 1, smax, ymin, ymax)

    smax = min(x_limit, y_limit, smax)

    if smax <= smin:
        square_sidelength = smax
    else:
        square_sidelength = np.random.randint(int(smin), int(smax))

    x1,x2 = corner_bounds(true_width - 1, square_sidelength, xmin, xmax)
    y1,y2 = corner_bounds(true_height- 1, square_sidelength, ymin, ymax)

    if x2 <= x1:
        square_x = x1
    else:
        square_x = np.random.randint(x1, x2)
        
    if y2 <= y1:
        square_y = y1
    else:
        square_y = np.random.randint(y1, y2)

    keypoints = keypoints - np.array([[float(square_x), float(square_y)]])

    return keypoints, (square_x, square_y, square_sidelength)

def focus_and_augment(img, keypoints, heatmap_targeter, num_cascades, OUTPUT_SIZE = 256, FOREHEAD_SCALE = 1.4, NORMALIZE = True,
                ROTATION_MAX = 10, FACE_FRAME_RATIO = 0.55, FROM_PREPROCESSED = True):

    true_height, true_width = img.shape[:2]

    img, keypoints = rotate_img(ROTATION_MAX, true_width, true_height, img, keypoints)

    ((xmin, ymin), (xmax, ymax)) = np.amin(keypoints.astype(int), 0), np.amax(keypoints.astype(int) + 1, 0)

    ymin = max(0, int(ymax - FOREHEAD_SCALE * (ymax - ymin)))
    height = ymax - ymin
    width = xmax - xmin
    
    keypoints, (sx, sy, length) = stochastic_padding(keypoints, height, width, FACE_FRAME_RATIO, 
                xmin, ymin, xmax, ymax, true_width, true_height)

    img = img[sy:sy+length, sx:sx+length]

    scale = OUTPUT_SIZE/length 

    img = cv2.resize(img, (OUTPUT_SIZE,OUTPUT_SIZE), interpolation = cv2.INTER_NEAREST)

    keypoints = keypoints * scale

    gaussians = heatmap_targeter(keypoints).astype('float32')

    return img, gaussians, keypoints

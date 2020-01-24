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
        self.std = std
        self.target_size = image_size
        self.distribution = multivariate_normal(cov=[[std**2, 0],[0, std**2]])

    def __call__(self, targets):
        targets = targets * self.heatmap_size/self.target_size
        num_points, coors = targets.shape
        assert(coors == 2), 'Array must be of size (num_points, 2)'
        x_targs, y_targs = targets[:, 0][np.newaxis, :], targets[:,1][np.newaxis,:]

        x = np.arange(self.heatmap_size)[:,np.newaxis]
        x_errors = np.expand_dims((x - x_targs).T, -2) #(194,64, 1)
        xx = np.tile(x_errors, (self.heatmap_size, 1))

        y = np.arange(self.heatmap_size)[:,np.newaxis]
        y_errors = np.expand_dims((y - y_targs).T, -1)
        yy = np.tile(y_errors, self.heatmap_size)

        errors = np.concatenate([np.expand_dims(xx, -1), np.expand_dims(yy, -1)], axis = -1)
        
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
    elif xmax - square_sidelength <= 0:
        square_x = 0
    else:
        square_x = np.random.randint(xmax - square_sidelength, xmin)

    if ymax - square_sidelength >= ymin:
        square_y = max(ymax - square_sidelength, 0)
    elif ymax - square_sidelength <= 0:
        square_y = 0
    else:
        square_y = np.random.randint(ymax - square_sidelength, ymin)

    keypoints = keypoints - np.array([[float(square_x), float(square_y)]])

    return keypoints, (square_x, square_y, square_sidelength)

def rescale_image_keypoints(max_dimension, img, keypoints):
    
    height, width = img.shape[:2]

    scale = max_dimension / max(height, width)

    img = cv2.resize(img, (int(width * scale), int(height * scale)))

    keypoints = keypoints * scale

    return img, keypoints

def get_annotation_and_image(annotation_filename, image_dir):

    with open(annotation_filename) as f:
        corresponding_jpg = f.readline().strip()

    keypoints = np.loadtxt(annotation_filename, dtype = np.float32, delimiter=',',skiprows=1)
    
    if keypoints.shape != (194,2):
        raise AssertionError('Keypoints file contains wrong number of points!: {}'.format(annotation_filename))
    
    image_path = os.path.join(image_dir, corresponding_jpg + '.jpg')
    assert(os.path.exists(image_path))

    img = cv2.imread(image_path)

    return img, keypoints

def load_processed_sample(keypoint_filepath):

    keypoints = np.load(keypoint_filepath)

    dir_chain = os.path.normpath(keypoint_filepath).split(os.sep)

    jpg_name = os.path.basename(keypoint_filepath)[:-4] + '.jpg'

    jpg_path = os.path.join(*dir_chain[:-2], 'images', jpg_name)

    assert(os.path.exists(jpg_path))

    image = cv2.imread(jpg_path)

    return image, keypoints

def prepare_input(annotation_filename, heatmap_targeter, OUTPUT_SIZE = 256, image_dir = None, FOREHEAD_SCALE = 1.4, NORMALIZE = True,
                ROTATION_MAX = 10, FACE_FRAME_RATIO = 0.55, FROM_PREPROCESSED = True):

    if FROM_PREPROCESSED:
        img, keypoints = load_processed_sample(annotation_filename)
    else:
        assert(not image_dir is None)
        img, keypoints = get_annotation_and_image(annotation_filename, image_dir)


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

    #augment brightness, contrast, and other stuff here

    #normalize to [-0.5, 0.5], [-1,1]
    if NORMALIZE:
        normalized_image = ((img / 255.) - 1.).astype(np.float32)
        normalized_keypoints = ((2. * keypoints / OUTPUT_SIZE) - 1.).astype(np.float32)
    else:
        normalized_image = img
        normalized_keypoints = keypoints

    gaussians = heatmap_targeter(keypoints)

    print('gaussina hsape:', gaussians.shape)

    return normalized_image, (gaussians.astype(np.float32), normalized_keypoints)

class LandmarkImageGenerator():

    def __init__(self, preprocessed_datadir, input_image_size, heatmap_size, heatmap_std, **kwargs):

        self.heatmapper = HeatmapTargeter(input_image_size, heatmap_size, heatmap_std)

        keypoints_dir = os.path.join(preprocessed_datadir, 'keypoints')
        self.files = [os.path.join(keypoints_dir, filename) for filename in os.listdir(keypoints_dir)]
        self.i = 0
        self.prepare_input_kwargs = kwargs


    def __call__(self):

        while True:

            keypoints_filepath = self.files[self.i]

            yield prepare_input(keypoints_filepath, self.heatmapper, **self.prepare_input_kwargs)

            self.i += 1

            if self.i >= len(self.files):
                self.i = 0


if __name__ == "__main__":
    #TEST_ANNOTATION = './HELEN/annotation/2.txt'
    #IMAGE_DIR = './HELEN/processed_samples/images'
    #ANNOTATION_DIR = './HELEN/processed_samples/keypoints'

    samples_dir ='./HELEN/processed_samples/'

    gen = iter(LandMarkImageGenerator(samples_dir, 256, 64, 2.5, NORMALIZE = False)())

    for i in range(10):
        
        image, (keypoints, gaussians) = next(gen)

        cv2.imshow('im', show_keypoints(image, keypoints))
        cv2.waitKey()


# %%

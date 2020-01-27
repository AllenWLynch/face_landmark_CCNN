
#%%
import findspark
findspark.init()

import pyspark
sc = pyspark.SparkContext(appName="ReduceImageSize")
#%%
import os
from data_utils import HeatmapTargeter, focus_and_augment
import numpy as np
import cv2

IMAGE_DIR = './HELEN/train'
ANNOTATION_DIR = './HELEN/annotation'
SAVEDIR = './HELEN/processed_samples'

annotation_files = [os.path.join(ANNOTATION_DIR, filename) for filename in os.listdir(ANNOTATION_DIR)]
img_files = [os.path.join(IMAGE_DIR, filename) for filename in os.listdir(IMAGE_DIR)]

# %%

img_files_rdd = sc.parallelize(img_files, 10)
annotation_files = sc.parallelize(annotation_files, 10)

#%%
#1 read annotation files
def parse_annotation(annotation_filename):

    with open(annotation_filename) as f:
        corresponding_jpg = f.readline().strip()

    keypoints = np.loadtxt(annotation_filename, dtype = np.float32, delimiter=',',skiprows=1)
    
    return corresponding_jpg + '.jpg', keypoints

annotation_data = annotation_files.map(parse_annotation) #(jpg_filename, keypoints)

#%%
#open and rescale images to a managable size
def open_and_rescale(image_filename, max_dim):

    img = cv2.imread(image_filename)
    assert(not img is None)

    height, width = img.shape[:2]

    scale = max_dim / max(height, width)

    img = cv2.resize(img, (int(width * scale), int(height * scale)))

    return os.path.basename(image_filename), (scale, img)

scaled_images = img_files_rdd.map(lambda x : open_and_rescale(x, 512)) #(filename, (scale, image))

#%%
#join images and their annotation
joined = scaled_images.join(annotation_data) #(filename, ((scale, image), keypoints))

joined.persist()

all_info = joined.collect()

assert(False)

#%%
#scale the keypoints
def scale_kpts(K,W):

    (scale, img) = K
    keypoints = W

    return img, keypoints * scale

all_scaled = scaled_images.map(lambda x : scale_kpts(x[1])) #(img, keypoints)

#%%
#now create the dataset by taking three random crops of each sample

def augment(x):

    heatmap_targeter = HeatmapTargeter(256, 64, 2.5)

    return [
        focus_and_augment(img, keypoints, heatmap_targeter, 3)
        for (img, keypoints) in x
    ]

augments = all_scaled.glom().flatMap(augment)

samples = augments.collect()

# %%

#%%
import findspark
findspark.init()

import pyspark
sc = pyspark.SparkContext(appName="ReduceImageSize")
#%%
import os
from data_utils import get_annotation_and_image, rescale_image_keypoints
import numpy as np
import cv2

IMAGE_DIR = './HELEN/train'
ANNOTATION_DIR = './HELEN/annotation'
SAVEDIR = './HELEN/processed_samples'
files = [os.path.join(ANNOTATION_DIR, filename) for filename in os.listdir(ANNOTATION_DIR)]

# %%

rdd = sc.parallelize(files)

def process_file(annotation_filename, max_dimension_size, image_dir, savedir):

    image, keypoints = get_annotation_and_image(annotation_filename, image_dir)

    image, keypoints = rescale_image_keypoints(max_dimension_size, image, keypoints)

    newfilename = os.path.basename(annotation_filename)[:-4]

    cv2.imwrite(os.path.join(savedir, 'images', newfilename + '.jpg'), image)
    np.save(os.path.join(savedir, 'keypoints', newfilename), keypoints)

    return True


successes = rdd.map(lambda x : process_file(x, 750, IMAGE_DIR, SAVEDIR))

successes = successes.map(lambda x : (x,1))

successes = successes.countByKey()

print('Converted {} images.'.format(successes[True]))

if sc:
    sc.stop()

# %%

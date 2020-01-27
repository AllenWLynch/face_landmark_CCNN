import data_utils
import os
import cv2
import numpy as np

DATADIR = './HELEN/processed_samples/test'
DEST_DIR = './HELEN/HELEN_dataset/train'
NUMREPS = 3

files = [os.path.join(DATADIR, 'images', filename) for filename in os.listdir(os.path.join(DATADIR, 'images'))]

heatmapper = data_utils.HeatmapTargeter(256,64,2.5)

samples_generated = 6390
for i, image_filepath in enumerate(files):

    print('\rStep: ' + str(i), end = '')

    basename = os.path.basename(image_filepath)[:-4]
    keypoint_filepath = os.path.join(DATADIR, 'keypoints',basename)

    img = cv2.imread(image_filepath)
    assert(not img is None)

    keypoints = np.load(keypoint_filepath + '.npy').astype('float32')
    
    samples = [
        data_utils.focus_and_augment(img, keypoints, heatmapper, 3)
        for i in range(NUMREPS)
    ]

    for sample in samples:
        samples_generated += 1
        cv2.imwrite(os.path.join(DEST_DIR, 'images', str(samples_generated) + '.jpg'), sample[0])
        np.save(os.path.join(DEST_DIR, 'keypoints', str(samples_generated)), sample[2])
        np.save(os.path.join(DEST_DIR, 'heatmaps', str(samples_generated)), sample[1])



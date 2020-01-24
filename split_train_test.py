


import os
import numpy as np

TRAIN_DIR = './HELEN/processed_samples/train'
TEST_DEST = './HELEN/processed_samples/test'

TEST_SIZE = 200

num_files = len(os.listdir(os.path.join(TRAIN_DIR, 'keypoints')))

random_selections = np.random.choice(np.arange(num_files), 200)

for selection in random_selections:

    #transfer file

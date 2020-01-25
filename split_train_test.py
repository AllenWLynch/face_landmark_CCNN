


import os
import numpy as np

TRAIN_DIR = './HELEN/processed_samples/train'
TEST_DEST = './HELEN/processed_samples/test'

TEST_SIZE = 200

num_files = len(os.listdir(os.path.join(TRAIN_DIR, 'keypoints')))

random_selections = np.random.choice(np.arange(num_files), 11)

moved= 0

for selection in random_selections:
    if os.path.exists(os.path.join(TRAIN_DIR, 'images', str(selection) + '.jpg')):
        os.rename(os.path.join(TRAIN_DIR, 'images', str(selection) + '.jpg'), os.path.join(TEST_DEST, 'images', str(selection) + '.jpg'))
        os.rename(os.path.join(TRAIN_DIR, 'keypoints', str(selection) + '.npy'), os.path.join(TEST_DEST, 'keypoints/', str(selection) + '.npy'))
        print('Moved sample #{}'.format(str(selection)))
        moved += 1
    
print('Moved {} samples'.format(moved))


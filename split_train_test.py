


import os
import numpy as np

TRAIN_DIR = './HELEN/HELEN_dataset/train'
TEST_DEST = './HELEN/HELEN_dataset/test'

TEST_SIZE = 300

num_files = len(os.listdir(os.path.join(TRAIN_DIR, 'images')))

random_selections = np.random.choice(np.arange(num_files), TEST_SIZE)

moved= 0

for selection in random_selections:
    if os.path.exists(os.path.join(TRAIN_DIR, 'images', str(selection) + '.jpg')):
        os.rename(os.path.join(TRAIN_DIR, 'images', str(selection) + '.jpg'), os.path.join(TEST_DEST, 'images', str(selection) + '.jpg'))
        os.rename(os.path.join(TRAIN_DIR, 'keypoints', str(selection) + '.npy'), os.path.join(TEST_DEST, 'keypoints/', str(selection) + '.npy'))
        os.rename(os.path.join(TRAIN_DIR, 'heatmaps', str(selection) + '.npy'), os.path.join(TEST_DEST, 'heatmaps/', str(selection) + '.npy'))
        print('Moved sample #{}'.format(str(selection)))
        moved += 1
    
print('Moved {} samples'.format(moved))


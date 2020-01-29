#%%
import tensorflow as tf
import numpy as np
import cv2
import networks
import data_utils
import training_utils
import os
import matplotlib.pyplot as plt

#%%
TRAIN_DIR = './HELEN/HELEN_dataset/train'
TEST_DIR = './HELEN/HELEN_dataset/test'


#%%
NUM_FEATURES = 194
IMG_SHAPE = (256,256,3)

rccnn = networks.RCCNN(NUM_FEATURES, IMG_SHAPE)

LOAD_FROM_CHECKPOINT = False
CHECKPOINT_DIR = './checkpoints'

checkpoint = tf.train.Checkpoint(landmark_cascade = rccnn)

try:
    checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR)).assert_consumed()
except Exception as err:
    print('Failed to load models from checkpoint!')
    raise err

#%%

test_img = cv2.imread(os.path.join(TEST_DIR, 'images', '17.jpg'))

train_img = cv2.imread(os.path.join(TRAIN_DIR,'images','1.jpg'))

#%%

def show_image(img):
    winname = 'Test'
    #img = data_utils.show_keypoints(img, kpts[2])
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
    cv2.imshow(winname, img)
    cv2.waitKey()
    cv2.destroyAllWindows()



# %%

norm_img = (test_img/255)- 0.5

# %%
heatmaps, landmarks = rccnn(np.array([norm_img]))

# %%
investigate = 15

keypoints = (landmarks[0] + 1.) * 128.

keypoints = keypoints + np.array([[[0,0]],[[256,0]], [[512,0]]])

keypoints = np.concatenate(keypoints[:], axis = 0)

concat = np.concatenate([test_img, test_img, test_img], axis = 1)

#%%
def show_keypoints(img, kpts, color = [0,255,0]):
    kp_img = np.copy(img)
    height, width = img.shape[:2]
    int_keypoints = kpts.astype(int).T
    int_keypoints[0, :] = np.clip(int_keypoints[0, :], 0, width-1)
    int_keypoints[1, :] = np.clip(int_keypoints[1,:], 0, height-1)
    kp_img[int_keypoints[1],int_keypoints[0]] = color
    return kp_img

overlay = show_keypoints(concat, keypoints)

show_image(overlay)

# %%
cv2.imwrite('./examples/test_set_rd1_results.jpg', overlay)

# %%

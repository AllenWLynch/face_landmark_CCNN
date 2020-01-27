#%%
import tensorflow as tf
import networks
import data_utils
import cv2
import training_utils
import numpy as np

#%%

NUM_FEATURES = 194
IMG_SHAPE = (256,256,3)

rccnn = networks.RCCNN(NUM_FEATURES, IMG_SHAPE)


#%%

CHECKPOINT_DIR = './checkpoints'

latest = tf.train.latest_checkpoint(CHECKPOINT_DIR)

# %%
rccnn.load_weights('./checkpoints/ckpt_50')

# %%

X,Y = training_utils.load_HELEN_dataset('./HELEN/HELEN_dataset/train', 3, 0.1)

#%%
INDEX = 13
heatmap, points = rccnn(np.array([X[INDEX]]))

kpts = ((points[0] + 1) * 128).numpy()
img = (X[INDEX] + 1.) * 255

winname = 'Test'
img = data_utils.show_keypoints(img, kpts[2])
cv2.namedWindow(winname)        # Create a named window
cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
cv2.imshow(winname, img)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
cv2.imwrite('./examples/training_loop_1.png',img)

# %%

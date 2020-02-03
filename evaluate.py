#%%
import tensorflow as tf
import numpy as np
import cv2
import networks
import data_utils
import training_utils
import os
import matplotlib.pyplot as plt
from collections import namedtuple
import importlib
import seaborn
import pandas as pd
import random
#%%
training_utils = importlib.reload(training_utils)

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
TRAIN_DIR = './HELEN/HELEN_dataset/train'
TEST_DIR = './HELEN/HELEN_dataset/test'

dataset_args = ((tf.float32, (tf.float32, tf.float32)),
                    (tf.TensorShape([256,256,3]), (tf.TensorShape([64,64,194]), tf.TensorShape([194,2]))), 
                    training_utils.FLL_preprocces(3), 
                    4, 5)

train_dataset = training_utils.TFGenerator(training_utils.KeypointsDataset(TRAIN_DIR), *dataset_args)
                    
test_dataset = training_utils.TFGenerator(training_utils.KeypointsDataset(TEST_DIR), *dataset_args)

#%%

def show_image(img):
    winname = 'Test'
    #img = data_utils.show_keypoints(img, kpts[2])
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
    cv2.imshow(winname, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


#%%

Result = namedtuple('result', ['image', 'predicted_landmarks', 'MSE', 'heatmap_error'])

def getResults(model, dataset, num_samples):

    results = []
    iters = num_samples//4
    for images, (heatmaps, landmarks) in dataset.take(iters):

        #m, c, k, 2
        pred_heatmap, pred_landmarks = model(images)

        pred_landmarks = pred_landmarks.numpy()

        unnorm_image = images.numpy() + 0.5
        unnorm_keypoints = 128. * (pred_landmarks + 1.)

        #m, c, k
        squared_error = np.square(pred_landmarks - landmarks).mean(axis = -1)

        heatmap_error = tf.keras.losses.categorical_crossentropy(heatmaps, pred_heatmap).numpy().mean(axis = (-1,-2))

        for datapoint in zip(unnorm_image, unnorm_keypoints, squared_error, heatmap_error):
            results.append(Result(*datapoint))
        
    return results

test_results = getResults(rccnn, test_dataset, 300)
train_results = getResults(rccnn, train_dataset, 300)


#%%

heatmap_error = np.array(list(zip(*test_results))[3])
heatmap_error.shape
#%%
df = pd.DataFrame(heatmap_error, columns = ['1','2','3']).reset_index()

df = pd.melt(df, id_vars = ['index'], var_name='Cascade', value_name = 'CE loss')
df

#%%
fig = seaborn.catplot(x = 'Cascade',y='CE loss',data = df, kind ='point')
fig.savefig('readme_materials/ce_loss.png')
#%%
if False:
    #%%
    #1. make image square
#%%
square = np.empty((9,256,256,3))
for i in range(9):
    img = test_results[i][0]
    keypoints = test_results[i][1][-1]

    overlay = data_utils.show_keypoints(img, keypoints, color =[0.,1.,0.])
    show_image(overlay)
    square[i] = overlay

#%%
big_square = np.concatenate([np.concatenate(square[i:i+3], axis = 1) for i in range(0,9,3)], axis = 0)
show_image(big_square)
#%%

plt.imshow(big_square)
#%%
cv2.imwrite('readme_materials/test_image_square.jpg',255 * big_square)

#%%
    #2. Find per-keypoint loss
    keypoint_losses = np.array(list(zip(*results))[2]).mean(axis = -1).T.mean(axis = -1)
    keypoint_losses.shape
    #%%
    keypoint_positions = np.array(list(zip(*results))[1]).mean(axis = 0)
    keypoint_positions.shape

    #%%
    seaborn.set()

    ax = seaborn.scatterplot(x = keypoint_positions[:,0], y = 256 - keypoint_positions[:,1], size = 5*keypoint_losses, legend=False)
    ax.set_yticks([])
    ax.set_xticks([])

    ax.figure.savefig('per_landmark_error.png')

    #%%
    #3. Train vs. Test error
    test_se = np.array(list(zip(*test_results))[2])[:,-1,:].mean(axis = -1).reshape(-1)
    train_se = np.array(list(zip(*train_results))[2])[:,-1,:].mean(axis = -1).reshape(-1)

    print(test_se.shape)
    #%%
    df = pd.DataFrame({'Test' : test_se, 'Train' : train_se}, index = np.arange(300))
    df.reset_index(inplace = True)
    df = pd.melt(df, id_vars = ['index'], var_name='Dataset', value_name = 'MSE')

    df
    #%%

    df = df.drop(df[df.MSE > 0.00125].index)
    #%%
    ax = seaborn.catplot(x = 'Dataset', y = 'MSE', data = df, kind ='violin')

    #%%

    test_error_recurrent = np.array(list(zip(*test_results))[2])
    test_error_recurrent = np.transpose(test_error_recurrent, axes = [1,0,2])
    test_error_recurrent.shape
    #%%
    per_recurrent_unit_mse = test_error_recurrent.mean(axis = -1)
    per_recurrent_unit_mse.shape

    #%%
    df = pd.DataFrame(data = {'1': per_recurrent_unit_mse[0,:], '2' : per_recurrent_unit_mse[1,:], '3' : per_recurrent_unit_mse[2,:]})
    df
    #%%
    df.reset_index(inplace = True)
    df = pd.melt(df, id_vars=['index'], var_name ='Cascade',value_name='MSE')
    df
    #%%
    fig = seaborn.catplot(x = 'Cascade', y = 'MSE', data = df, kind = 'point')
    fig.savefig('cascades_mse.png')


#%%




#%%
def show_keypoints(img, kpts, color = [0,255,0]):
    kp_img = np.copy(img)
    height, width = img.shape[:2]
    int_keypoints = kpts.astype(int).T
    int_keypoints[0, :] = np.clip(int_keypoints[0, :], 0, width-1)
    int_keypoints[1, :] = np.clip(int_keypoints[1,:], 0, height-1)
    kp_img[int_keypoints[1],int_keypoints[0]] = color
    return kp_img


# %%

for i in range(20):
    r = random.randint(1,300)
    print(r)
    img, landmarks, errors = test_results[r]
    landmarks.shape

    keypoints = landmarks[-1]

    keypoints = keypoints + np.array([[[0,0]],[[256,0]], [[512,0]]])

    keypoints = np.concatenate(keypoints[:], axis = 0)

    concat = np.concatenate([img, img, img], axis = 1)

    overlay = show_keypoints(concat, keypoints)

    show_image(overlay)

# %%

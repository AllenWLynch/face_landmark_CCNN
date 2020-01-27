#%%
import tensorflow as tf
import numpy as np
import cv2
from data_utils import show_keypoints
import os
import datetime
from random import shuffle


class FLDRegressionCallback(tf.keras.callbacks.Callback):

    def __init__(self, save_dir, test_datagen):
        newfilename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.save_dir = os.path.join(save_dir, newfilename)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.test_datagen = test_datagen

    def on_epoch_begin(self, epoch, logs = None):
        if epoch <= 1:
            self.on_epoch_end(epoch - 1, logs)
    
    def on_epoch_end(self, epoch, logs = None):
        
        img, (heatmaps, keypoints) = next(iter(self.test_datagen()))

        heatmap, regression = self.model(np.array([img]))

        regression = regression[0][-1]

        coordinates = (regression + 1.) * 128. 
        coordinates = np.clip(coordinates, 0, 256)

        unnorm_img = np.clip((255 * (img + 1.)).astype(int), 0, 255)

        img = show_keypoints(unnorm_img, coordinates)

        cv2.imwrite(os.path.join(self.save_dir, 'epoch_{}.jpg'.format(str(epoch))), img)

def load_HELEN_dataset(DATADIR, cascades, proportion):

    files = [os.path.join(DATADIR, 'images', filename) for filename in os.listdir(os.path.join(DATADIR, 'images'))]
    shuffle(files)

    images = []
    heatmaps = []
    keypoints = []

    for i, filepath in enumerate(files[:int(proportion * len(files))]):

        print('\rRead in {} files'.format(str(i)), end = '')
        image_filepath = filepath
        image = cv2.imread(image_filepath)

        basename = os.path.basename(image_filepath)[:-4] + '.npy'

        heatmap = np.load(os.path.join(DATADIR, 'heatmaps', basename))
        kpts = np.load(os.path.join(DATADIR, 'keypoints', basename))

        heatmap = np.expand_dims(heatmap, 0)
        heatmap = np.tile(heatmap, (cascades, 1,1,1))

        kpts = np.expand_dims(kpts, 0)
        kpts = np.tile(kpts, (cascades, 1,1))

        image = (image / 255.) - 1.
        kpts = (kpts / 128.) - 1.

        images.append(image)
        heatmaps.append(heatmap)
        keypoints.append(kpts)

    heatmaps = np.array(heatmaps).astype('float32')
    keypoints = np.array(keypoints).astype('float32')
    images = np.array(images).astype('float32')

    return images, (heatmaps, keypoints)


class KeypointsDataset():

    def __init__(self, dir):
        self.dir = dir
        self.subdirs = {'images' : os.path.join(dir ,'images'),
                        'keypoints' : os.path.join(dir, 'keypoints'),
                        'heatmaps' : os.path.join(dir, 'heatmaps')}

        self.filepaths = self.list_filepaths(self.subdirs['images'])
        shuffle(self.filepaths)

    
    @staticmethod
    def list_filepaths(dir):
        return [os.path.join(dir, filename) for filename in os.listdir(dir)]

    def __call__(self):
        
        i = 0
        while True:

            image_filepath = self.filepaths[i]
            image = cv2.imread(image_filepath)

            basename = os.path.basename(image_filepath)[:-4] + '.npy'

            heatmaps = np.load(os.path.join(self.subdirs['heatmaps'], basename))
            keypoints = np.load(os.path.join(self.subdirs['keypoints'], basename))

            yield image, heatmaps, keypoints

            i+=1
            if i >= len(self.filepaths):
                i = 0
                shuffle(self.filepaths)
        
class FLL_preprocces():

    def __init__(self, num_cascades):
        self.cascades = num_cascades

    def __call__(self, x,y):

        image, (heatmap, keypoints) = x,y
        
        image = (image / 255.) - 1.
        keypoints = (keypoints / 128.) - 1

        keypoints = tf.expand_dims(keypoints, 0)
        heatmap = tf.expand_dims(heatmap, 0)
        
        keypoints = tf.tile(keypoints, [self.cascades, 1, 1])
        heatmap = tf.tile(heatmap, [self.cascades, 1,1,1])

        image = tf.cast(image, tf.float32)
        keypoints = tf.cast(keypoints, tf.float32)
        heatmap = tf.cast(heatmap, tf.float32)

        return image, (heatmap, keypoints)

def TFGenerator(python_generator, datatypes, shapes, tf_preprocessing_fn, batch_size, prefetch_num):

    dataset = tf.data.Dataset.from_generator(python_generator, datatypes, shapes)
    dataset = dataset.map(tf_preprocessing_fn)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(prefetch_num)

    return dataset

if __name__ == "__main__":

    DIR = './HELEN/HELEN_dataset/train'

    X,Y = load_HELEN_dataset(DIR, 3, 0.05)

    winname = 'Test'
    img = show_keypoints(image, kpts)
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
    cv2.imshow(winname, img)
    cv2.waitKey()
    cv2.destroyAllWindows()
# %%

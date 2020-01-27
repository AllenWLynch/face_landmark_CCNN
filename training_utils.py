#%%
import tensorflow as tf
import numpy as np
import cv2
from data_utils import show_keypoints
import os
import datetime
from random import shuffle


class FLLTrainer():

    def __init__(self, model, optmizer, heatmap_loss_obj, regression_loss_obj, logger, 
            heatmap_metrics, landmark_metrics, logger_steps = 50):
        self.model = model
        self.optim = optmizer
        self.hm_loss = heatmap_loss_obj
        self.regression_loss = regression_loss_obj
        self.logger = logger
        self.logger_steps = logger_steps
        self.train_steps = 0
        self.epoch_steps = 0
        self.heatmap_metrics = heatmap_metrics
        self.landmark_metrics = landmark_metrics

    @tf.function()
    def _train_step(self, image, heatmap, landmarks):
        with tf.GradientTape() as tape:

            heatmap_prediction, regression_prediction = self.model(image, training = True)

            heatmap_loss = self.hm_loss(heatmap, heatmap_prediction)
            regression_loss = self.regression_loss(landmarks, regression_prediction)

            loss = heatmap_loss + regression_loss

            scaled_loss = self.optim.get_scaled_loss(loss)

        scaled_gradients  = tape.gradient(loss, self.model.trainable_weights)
        grads = self.optim.get_unscaled_gradients(scaled_gradients)
        self.optim.apply_gradients(zip(grads, self.model.trainable_weights))

        return loss, heatmap_loss, regression_loss

    def train_epoch(self, steps, dataset):

        for i, (image, (heatmap, landmarks)) in enumerate(dataset.take(steps)):

            print('\rStep {}'.format(str(i + 1)), end = '')

            loss, heatmap_loss, regression_loss = self._train_step(image, heatmap, landmarks)

            if i % self.logger_steps == 0:
                with self.logger.as_default():
                    tf.summary.scalar('Heatmap Loss', heatmap_loss, step=self.train_steps)
                    tf.summary.scalar('Regression Loss', regression_loss, step=self.train_steps)
                    tf.summary.scalar('Total Loss', loss, step=self.train_steps)
            self.train_steps = self.train_steps + 1
        print('')

    
    @tf.function()
    def _test_step(self, image):
        return self.model(image, training = False)

    def evaluate(self, steps, dataset, num_examples):

        examples = []
        for i, (image, (heatmap, landmarks)) in enumerate(dataset.take(steps)):

            print('\rValidation Step {}'.format(str(i+1)), end = '')

            heatmap_prediction, regression_prediction = self._train_step(image)

            for metric in self.heatmap_metrics:
                metric(heatmap, heatmap_prediction)

            for metric in self.landmark_metrics:
                metric(landmarks, regression_prediction)

            if len(examples) < num_examples:
                examples.append((image[0], regression_prediction.numpy()[0][2]))

        #manipulate examples
        display_images = []
        for (img, keypoints) in examples:
            
            #un-normalize
            img = np.array((img + 0.5) * 255).astype('uint8')
            keypoints = (keypoints + 1.) * 128

            display_image = show_keypoints(img, keypoints)
            display_images.append(display_image)

        display = np.expand_dims(np.concatenate(display_images, axis = 1),0)

        with self.logger.as_default():
            tf.summary.image('Examples', display, step = self.epoch_steps)
            for metric in self.heatmap_metrics:
                tf.summary.scalar('Heatmap ' + metric.name, metric.result(), step = self.epoch_steps)
            for metric in self.landmark_metrics:
                tf.summary.scalar('Landmark ' + metric.name, metric.result(), step = self.epoch_steps)
        print('')
        self.epoch_steps = self.epoch_steps + 1

    def fit(self, train_dataset, test_dataset, epochs, steps_per_epoch, evaluation_steps, checkpoint_manager, checkpoint_every):

        try:
            for epoch in range(epochs):
                print('EPOCH ', epoch + 1)
                
                self.train_epoch(steps_per_epoch, train_dataset)

                self.evaluate(evaluation_steps, test_dataset, 3)        

                if (epoch + 1) % checkpoint_every == 0:
                    checkpoint_manager.save()
                    print('Saved Checkpoint!')    

        except KeyboardInterrupt:
            print('Training interupted!')
            user_input = ''
            while not (user_input == 'y' or user_input == 'n'):
                user_input = input('Save model\'s current state?: [y/n]')
            if user_input == 'y':
                checkpoint_manager.save()
                print('Saved checkpoint!')
            
        else:
            print('Training complete! Saving final model.')
            checkpoint_manager.save()


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

        image = (image / 255.) - 0.5
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
    
    @staticmethod
    def list_filepaths(dir):
        return [os.path.join(dir, filename) for filename in os.listdir(dir)]

    def __call__(self):
        
        shuffle(self.filepaths)

        i = 0
        while True:

            image_filepath = self.filepaths[i]
            image = cv2.imread(image_filepath)

            basename = os.path.basename(image_filepath)[:-4] + '.npy'

            heatmaps = np.load(os.path.join(self.subdirs['heatmaps'], basename))
            keypoints = np.load(os.path.join(self.subdirs['keypoints'], basename))

            yield image, (heatmaps, keypoints)

            i+=1
            if i >= len(self.filepaths):
                i = 0
                shuffle(self.filepaths)
        
class FLL_preprocces():

    def __init__(self, num_cascades):
        self.cascades = num_cascades

    @tf.function()
    def __call__(self, x, y):

        image = x
        (heatmap, keypoints) = y
        
        image = (image / 255.) - 0.5
        keypoints = (keypoints / 128.) - 1.

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


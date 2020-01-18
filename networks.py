import tensorflow as tf
import tensorflow.keras.layers as layers


class FeatureSoftmaxLayer(layers.Layer):

    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        (_, h, w, num_features) = input_shape
        self.transposer = layers.Permute((3,1,2))
        self.reshaper = layers.Reshape((num_features, h*w))
        self.softmaxer = layers.Softmax(axis = -1)
        self.unflattener = layers.Reshape((num_features, h, w))

    def call(self, X):

        X = self.transposer(X)
        X = self.reshaper(X)
        X = self.softmaxer(X)
        X = self.unflattener(X)
        return X

def ConvBNReluBlock(num_channels, filter_size, stride = 1, padding = 'SAME', **kwargs):
    return tf.keras.Sequential([
        layers.Conv2D(num_channels, filter_size, strides = stride, padding = padding, **kwargs),
        layers.BatchNormalization(),
        layers.ReLU()
    ])

def FeatureCNN(input_shape):
    return tf.keras.Sequential([
        ConvBNReluBlock(64,3, input_shape = input_shape),
        ConvBNReluBlock(64,3),
        layers.MaxPool2D(2),
        ConvBNReluBlock(128,3),
        ConvBNReluBlock(128,3),
        layers.MaxPool2D(2),
        ConvBNReluBlock(128,3),
        ConvBNReluBlock(128,3),
    ])

#maybe I want this to be a small unet
def HeatmapCNN(input_shape, num_features):
    return tf.keras.Sequential([
        ConvBNReluBlock(256,9, input_shape = input_shape),
        ConvBNReluBlock(512, 9),
        ConvBNReluBlock(256,1),
        ConvBNReluBlock(256,1),
        layers.Conv2D(num_features, 1),
        FeatureSoftmaxLayer(),
    ])

def CascadingHeatmapCNN(num_features, feature_shape, prev_heatmap_shape):

    features = tf.keras.Input(shape = feature_shape, name = 'Features')

    prev_heatmap = tf.keras.Input(shape = prev_heatmap_shape, name = 'Previous_heatmap')

    X = layers.Concatenate(axis = -1)(features, prev_heatmap)

    output_heatmap = HeatmapCNN(X.get_shape()[1:], num_features)(X)

    return tf.keras.Model((features, prev_heatmap), output_heatmap)

def RegressionFeatureCNN(input_shape):
    return tf.keras.Sequential([
        ConvBNReluBlock(64,8,stride=2, input_shape = input_shape),
        ConvBNReluBlock(128,6,stride=2),
        ConvBNReluBlock(256,3),
        layers.MaxPool2D(2)
    ])

def RegressionPredictionCNN(num_features, input_shape):
    return tf.keras.Sequential([
        ConvBNReluBlock(512, 4, stride = 2, input_shape = input_shape),
        ConvBNReluBlock(1024, 4, stride = 2),
        layers.MaxPool2D(2),
        layers.Conv2D(num_features, 1),
    ])

def CascadingRegressionCNN(num_features, feature_input_shape, heatmap_shape, prev_regression_feature_shape):

    features = tf.keras.Input(shape = feature_input_shape)
    heatmap = tf.keras.Input(shape = heatmap_shape)
    prev_regression = tf.keras.Input(shape = prev_regression_feature_shape)

    feature_input = layers.Concatenate(axis = -1)([features, heatmap])

    regression_features = RegressionFeatureCNN(feature_input.get_shape()[1:])(feature_input)

    regression_X = layers.Concatenate(axis = -1)([regression_features, prev_regression])

    regression_output = RegressionPredictionCNN(num_features, regression_X.get_shape()[1:])(regression_X)

    return tf.keras.Model((features, heatmap, prev_regression), (regression_features, regression_output))

#last implementation
def RecurrentCascadingCNN(num_features, feature_input_shape, prev_heatmap_shape, prev_regression_feature_shape):

    features = tf.keras.Input(shape = feature_input_shape)
    prev_heatmap = tf.keras.Input(shape = prev_heatmap_shape)
    prev_regression = tf.keras.Input(shape = prev_regression_feature_shape)

    heatmap = CascadingHeatmapCNN(num_features, feature_input_shape, prev_heatmap_shape)([features, prev_heatmap])

    regression_features, regression_output = CascadingRegressionCNN(num_features, feature_input_shape, 
            heatmap.get_shape()[1:], prev_regression_feature_shape)([features, heatmap, prev_regression])

    return tf.keras.Model((features, prev_heatmap, prev_regression), (heatmap, regression_features, regression_output))

def RCCNN(num_features, image_shape, num_cascades = 3):

    img = tf.keras.Input(shape = image_shape, name = 'Input_image')

    features = FeatureCNN(image_shape)(img)

    FEATURE_SHAPE = features.get_shape()[1:]

    heatmap = HeatmapCNN(features.get_shape()[1:], num_features)(features)

    HEATMAP_SHAPE = heatmap.get_shape()[1:]

    regression_input = layers.Concatenate(axis = -1)([features, heatmap])

    regression_features = RegressionFeatureCNN(regression_input.get_shape()[1:])(regression_input)

    REGRESSION_FEATURES_SHAPE = regression_features.get_shape()[1:]

    prediction = RegressionPredictionCNN(num_features, regression_features.get_shape()[1:])(regression_features)
    
    recurrent_predictions = []
    recurrent_predictions.append((heatmap, prediction))

    recurrent_block = RecurrentCascadingCNN(num_features, FEATURE_SHAPE, HEATMAP_SHAPE, REGRESSION_FEATURES_SHAPE)

    for i in range(num_cascades):
        heatmap, regression_features, prediction = recurrent_block(features, heatmap, regression_features)
        recurrent_predictions.append(heatmap, prediction)

    return tf.keras.Model(img, recurrent_predictions)


num_features = 20

img_shape = (256,256,3)

model = RCCNN(num_features, image_shape)
print(model.summary())

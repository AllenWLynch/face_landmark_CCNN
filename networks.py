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
        self.untransposer = layers.Permute((2,3,1))


    def call(self, X):

        X = self.transposer(X)
        X = self.reshaper(X)
        X = self.softmaxer(X)
        X = self.unflattener(X)
        X = self.untransposer(X)
        return X

def ConvBNReluBlock(num_channels, filter_size, stride = 1, padding = 'SAME', **kwargs):
    return tf.keras.Sequential([
        layers.Conv2D(num_channels, filter_size, strides = stride, padding = padding, **kwargs),
        layers.BatchNormalization(),
        layers.ReLU()
    ])

#might replace this with miniceptions
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
    ], name = 'FeatureCNN')


class MiniCeptionLayer(layers.Layer):

    def __init__(self, k, conv_kernels, output_channels):
        super().__init__()
        self.conv_kernels = conv_kernels
        self.output_channels = output_channels
        self.k = k

    def build(self, input_shape):
        nc = input_shape[-1]
        reduced_channels = nc // self.k
        self.paths = [
            tf.keras.Sequential([
                ConvBNReluBlock(reduced_channels, 1),
                ConvBNReluBlock(output_channels, kernel_size),
            ])
        for kernel_size, output_channels in zip(self.conv_kernels, self.output_channels)]
        self.concatenator = layers.Concatenate(axis = -1)


    def call(self, X):
        return self.concatenator([path(X) for path in self.paths])


class SelfAttnLayer(tf.keras.layers.Layer):

    def __init__(self, k = 8, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def build(self, input_shape):
        (m, h, w, nc) = input_shape
        assert(nc // self.k > 0)
        self.flattener = tf.keras.layers.Reshape((h*w, -1))
        self.deflattener = tf.keras.layers.Reshape((h, w, nc))
        self.convQ = tf.keras.layers.Conv2D(nc//self.k, (1,1), padding = 'SAME')
        self.convK = tf.keras.layers.Conv2D(nc//self.k, (1,1), padding = 'SAME')
        self.convV = tf.keras.layers.Conv2D(nc, (1,1), padding = 'SAME')
        self.gamma = tf.Variable(0., dtype = 'float32', trainable = True, name = 'gamma')

    def call(self, X):

        Q = self.convQ(X)
        K = self.convQ(X)
        V = self.convV(X)

        Q = self.flattener(Q)
        K = self.flattener(K)
        V = self.flattener(V)

        energies = tf.matmul(Q, K, transpose_b = True)

        alphas = tf.nn.softmax(energies, axis = -1)

        o = tf.matmul(alphas, V)

        o_2D = self.deflattener(o)
        
        bypass = self.gamma * o_2D + X

        return bypass
        

def HeatmapCNN(input_shape, num_features):
    return tf.keras.Sequential([
        ConvBNReluBlock(128,7, input_shape = input_shape),
        ConvBNReluBlock(128,13),
        ConvBNReluBlock(256, 1),
        layers.Conv2D(num_features, 1),
        FeatureSoftmaxLayer()
    ], name = 'HeatmapCNN')


def CascadingHeatmapCNN(num_features, feature_shape, prev_heatmap_shape):

    features = tf.keras.Input(shape = feature_shape, name = 'Features')

    prev_heatmap = tf.keras.Input(shape = prev_heatmap_shape, name = 'Previous_heatmap')

    X = layers.Concatenate(axis = -1)([features, prev_heatmap])

    output_heatmap = HeatmapCNN(X.get_shape()[1:], num_features)(X)

    return tf.keras.Model((features, prev_heatmap), output_heatmap, name = 'CascadingHCNN')

def RegressionFeatureCNN(input_shape):
    return tf.keras.Sequential([
        ConvBNReluBlock(64,7,input_shape = input_shape),
        layers.MaxPool2D(2),
        ConvBNReluBlock(128,5),
        layers.MaxPool2D(2),
        ConvBNReluBlock(256,3),
        layers.MaxPool2D(2),
    ], name = 'RFCNN')

def RegressionPredictionCNN(num_features, input_shape):
    return tf.keras.Sequential([
        ConvBNReluBlock(256, 3, input_shape = input_shape),
        layers.MaxPool2D(2),
        ConvBNReluBlock(512, 3),
        layers.MaxPool2D(2),
        ConvBNReluBlock(512, 1),
        layers.Flatten(),
        #layers.Conv2D(2*num_features, 1),
        layers.Dense(2*num_features),
        layers.Reshape((num_features, 2))
    ], name = 'OutCNN')

def CascadingRegressionCNN(num_features, feature_input_shape, heatmap_shape, prev_regression_feature_shape):

    features = tf.keras.Input(shape = feature_input_shape)
    heatmap = tf.keras.Input(shape = heatmap_shape)
    prev_regression = tf.keras.Input(shape = prev_regression_feature_shape)

    feature_input = layers.Concatenate(axis = -1)([features, heatmap])

    regression_features = RegressionFeatureCNN(feature_input.get_shape()[1:])(feature_input)

    regression_X = layers.Concatenate(axis = -1)([regression_features, prev_regression])

    regression_output = RegressionPredictionCNN(num_features, regression_X.get_shape()[1:])(regression_X)

    return tf.keras.Model((features, heatmap, prev_regression), (regression_features, regression_output), name = 'CascadingRCNN')

def RCCNN(num_features, image_shape, num_cascades = 3):

    img = tf.keras.Input(shape = image_shape, name = 'Input_image')
    #define features network (1)
    features = FeatureCNN(image_shape)(img)

    FEATURE_SHAPE = features.get_shape()[1:]

    #assert(FEATURE_SHAPE[:-1] == prior_shape[:-1]), 'Ouptut of feature network must concatenate with the priors for heatmap prediction network'

    #define recurrent heatmap module (2)
    #heatmap_rcnn = CascadingHeatmapCNN(num_features, FEATURE_SHAPE, prior_shape)

    heatmap = HeatmapCNN(FEATURE_SHAPE, num_features)(features)
    
    HEATMAP_SHAPE = heatmap.get_shape()[1:]

    regression_features_input = layers.Concatenate(axis = -1)([features, heatmap])

    #(3)
    regression_features = RegressionFeatureCNN(regression_features_input.get_shape()[1:])(regression_features_input)

    REGRESSION_FEATURES_SHAPE = regression_features.get_shape()[1:]

    #outputs so far: F, H1, R1
    #regression_feature_rccn takes as input: features and new heatmap
    #contains networks (4) and (5)
    heatmap_rcnn = CascadingHeatmapCNN(num_features, FEATURE_SHAPE, HEATMAP_SHAPE)
    regression_prediction_rcnn = CascadingRegressionCNN(num_features, FEATURE_SHAPE, HEATMAP_SHAPE, REGRESSION_FEATURES_SHAPE)
        
    heatmaps = []
    regressions = []

    for _ in range(num_cascades):
        heatmap = heatmap_rcnn([features, heatmap])
        regression_features, prediction = regression_prediction_rcnn([features, heatmap, regression_features])
        heatmaps.append(heatmap)
        regressions.append(regression_features)

    heatmap_output = tf.stack(heatmaps, axis = 1)
    regression_output = tf.stack(regressions, axis = 1)

    return tf.keras.Model(img, (heatmap_output, regression_output))

def priors_RCCNN(num_features, image_shape, prior_shape, num_cascades = 3):

    img = tf.keras.Input(shape = image_shape, name = 'Input_image')
    priors = tf.keras.Input(shape = prior_shape, name = 'Prior')
    #define features network (1)
    features = FeatureCNN(image_shape)(img)

    FEATURE_SHAPE = features.get_shape()[1:]

    assert(FEATURE_SHAPE[:-1] == prior_shape[:-1]), 'Ouptut of feature network must concatenate with the priors for heatmap prediction network'

    #define recurrent heatmap module (2)
    heatmap_rcnn = CascadingHeatmapCNN(num_features, FEATURE_SHAPE, prior_shape)

    heatmap = heatmap_rcnn([features, priors])
    
    HEATMAP_SHAPE = heatmap.get_shape()[1:]

    regression_features_input = layers.Concatenate(axis = -1)([features, heatmap])

    #(3)
    regression_features = RegressionFeatureCNN(regression_features_input.get_shape()[1:])(regression_features_input)

    REGRESSION_FEATURES_SHAPE = regression_features.get_shape()[1:]

    #outputs so far: F, H1, R1
    #regression_feature_rccn takes as input: features and new heatmap
    #contains networks (4) and (5)
    regression_prediction_rcnn = CascadingRegressionCNN(num_features, FEATURE_SHAPE, HEATMAP_SHAPE, REGRESSION_FEATURES_SHAPE)
        
    heatmaps = []
    regressions = []

    for _ in range(num_cascades):
        heatmap = heatmap_rcnn([features, heatmap])
        regression_features, prediction = regression_prediction_rcnn([features, heatmap, regression_features])
        heatmaps.append(heatmap)
        regressions.append(regression_features)

    heatmap_output = tf.stack(heatmaps, axis = 1)
    regression_output = tf.stack(regressions, axis = 1)

    return tf.keras.Model((img, priors), (heatmap_output, regression_output))


'''
num_features = 194
img_shape = (256,256, 3)
prior_shape = (64,64,num_features)

regression_input = (8,8,512)
#model = RegressionPredictionCNN(194, regression_input)
model = RCCNN(num_features, img_shape, prior_shape)
print(model.summary())'''
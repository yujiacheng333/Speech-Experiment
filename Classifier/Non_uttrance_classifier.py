import tensorflow as tf
from Classifier.arcface import CMFace
from tensorflow.python.keras import backend as k
eps = 1e-6

class GLN(tf.keras.Model):
    def __init__(self):
        super(GLN, self).__init__()
        self.ndim = None
        self.gamma = None
        self.beta = None

    def build(self, input_shape):
        chs = input_shape[-1]
        self.ndim = len(input_shape)
        if self.ndim == 4:

            self.gamma = self.add_variable(name="gamma",
                                           shape=[1, 1, 1, chs],
                                           initializer=tf.keras.initializers.Ones(),
                                           trainable=True)
            self.beta = self.add_variable(name="beta",
                                          shape=[1, 1, 1, chs],
                                          initializer=tf.keras.initializers.Zeros(),
                                          trainable=True)
        if self.ndim == 3:
            self.gamma = self.add_variable(name="gamma",
                                           shape=[1, 1, chs],
                                           initializer=tf.keras.initializers.Ones(),
                                           trainable=True)
            self.beta = self.add_variable(name="beta",
                                          shape=[1, 1, chs],
                                          initializer=tf.keras.initializers.Zeros(),
                                          trainable=True)

    def call(self, inputs, training=None, mask=None):
        if self.ndim == 4:
            target_axis = [1, 2, 3]
        if self.ndim == 3:
            target_axis = [1, 2]
        mean, var = tf.nn.moments(inputs, keep_dims=True, axes=target_axis)
        inputs = (inputs - mean) / k.sqrt(var + eps) * self.gamma + self.beta
        return inputs


class ConvBNRelu(tf.keras.Model):
    def __init__(self, filters, strides, kernel_size, padding="same"):
        super(ConvBNRelu, self).__init__()
        self.conv = tf.keras.layers.Conv1D(filters=filters, strides=strides, kernel_size=kernel_size, padding=padding, use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization(momentum=.9)
    def call(self, inputs, training=None, mask=None):
        inputs = tf.nn.relu(self.bn(self.conv(inputs), training))
        return inputs

class ConvBN(tf.keras.Model):
    def __init__(self, filters, strides=1, kernel_size=1, padding="same"):
        super(ConvBN, self).__init__()
        self.conv = tf.keras.layers.Conv1D(filters=filters, strides=strides, kernel_size=kernel_size, padding=padding, use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization(momentum=.9)
    def call(self, inputs, training=None, mask=None):
        inputs = self.bn(self.conv(inputs), training)
        return inputs


class Encoder(tf.keras.Model):
    def __init__(self, filters, kernel_size, enc_activation=tf.keras.activations.linear):
        super(Encoder, self).__init__()
        self.conv1d = tf.keras.layers.Conv1D(filters=filters,
                                             kernel_size=kernel_size,
                                             strides=kernel_size//2,
                                             padding="valid",
                                             use_bias=False,
                                             activation=enc_activation)

    def call(self, inputs, training=None, mask=None):
        inputs_weights = self.conv1d(inputs)
        return inputs_weights


class SeModel(tf.keras.Model):
    def __init__(self, input_chs, reduction=2):
        super(SeModel, self).__init__()
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(units=input_chs//reduction, use_bias=False)
        self.bn0 = tf.keras.layers.BatchNormalization(momentum=0.9)
        self.dense_weight = tf.keras.layers.Dense(units=input_chs, use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.9)


    def call(self, inputs, training=None, mask=None):
        data = inputs
        inputs = self.pool(inputs)
        inputs = self.dense(inputs)
        inputs = tf.nn.relu(self.bn0(inputs, training))
        inputs = self.bn1(self.dense_weight(inputs), training)
        inputs = tf.keras.activations.hard_sigmoid(inputs)
        return inputs[:, tf.newaxis, tf.newaxis] * data


class Dconv1D(tf.keras.Model):
    def __init__(self, kernel_size, strides, dilation):
        super(Dconv1D, self).__init__()
        self.conv = tf.keras.layers.DepthwiseConv2D(kernel_size=(kernel_size, 1),
                                                    strides=(strides, 1), padding="same", dilation_rate=dilation,
                                                    use_bias=False)
    def call(self, inputs, training=None, mask=None):
        inputs = self.conv(inputs[:, :, tf.newaxis])[:, :, 0]
        return inputs

class DpSConv(tf.keras.Model):
    def __init__(self, input_chs, expand_rate, kernel_size, dilation=1):
        super(DpSConv, self).__init__()
        self.conv_expand = ConvBN(filters=input_chs*expand_rate)
        self.d_conv = Dconv1D(kernel_size=kernel_size, dilation=dilation, strides=1)
        self.compress_conv = ConvBN(filters=input_chs)
        self.bn = tf.keras.layers.BatchNormalization(momentum=.9)
    def call(self, inputs, training=None, mask=None):
        res = inputs
        inputs = tf.nn.leaky_relu(self.conv_expand(inputs, training))
        inputs = tf.nn.leaky_relu(self.bn(self.d_conv(inputs), training))
        inputs = self.compress_conv(inputs, training)
        return inputs + res




class NUClassifier(tf.keras.Model):
    def __init__(self, num_spk):
        super(NUClassifier, self).__init__()
        self.enc = ConvBNRelu(filters=512, kernel_size=40, strides=20)
        self.bottle = tf.keras.layers.Conv1D(filters=128, kernel_size=1, padding="same")
        self.bottle0_0 = DpSConv(input_chs=128, kernel_size=6, expand_rate=3, dilation=1)
        self.bottle0_1 = DpSConv(input_chs=128, kernel_size=6, expand_rate=3, dilation=2)
        self.Conv1 = ConvBNRelu(filters=256, kernel_size=3, padding="same", strides=2)
        self.bottle1_0 = DpSConv(input_chs=256, kernel_size=6, expand_rate=3, dilation=1)
        self.bottle1_1 = DpSConv(input_chs=256, kernel_size=6, expand_rate=3, dilation=2)
        self.Conv2 = ConvBNRelu(filters=512, kernel_size=3, padding="same", strides=2)
        self.bottle2_0 = DpSConv(input_chs=512, kernel_size=6, expand_rate=3, dilation=1)
        self.bottle2_1 = DpSConv(input_chs=512, kernel_size=6, expand_rate=3, dilation=2)
        self.Conv3 = ConvBNRelu(filters=512, kernel_size=3, padding="same", strides=2)
        self.compress = tf.keras.layers.Conv1D(filters=256, kernel_size=1, padding="same", strides=1)

        self.face = CMFace(num_spk)

    def call(self, inputs, return_embedding=False, training=True, mask=None):
        if training:
            inputs, labels = inputs
        inputs = self.enc(inputs[..., tf.newaxis], training)
        inputs = self.bottle(inputs)
        inputs = self.bottle0_0(inputs, training)
        inputs = self.bottle0_1(inputs, training)
        inputs = self.Conv1(inputs, training)

        inputs = self.bottle1_0(inputs, training)
        inputs = self.bottle1_1(inputs, training)
        inputs = self.Conv2(inputs, training)

        inputs = self.bottle2_0(inputs, training)
        inputs = self.bottle2_1(inputs, training)
        inputs = self.Conv3(inputs, training)
        inputs = self.compress(inputs)
        if not return_embedding:
            if training:
                inputs = k.mean(inputs, axis=1)
                inputs = self.face([inputs, labels], training=True)
            else:
                inputs = k.mean(inputs, axis=1)
                inputs = self.face(inputs, training=False)
            return inputs
        else:
            return inputs

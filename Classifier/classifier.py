import tensorflow as tf
from Classifier.arcface import CMFace
from tensorflow.python.keras import backend as k
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
        elif self.ndim == 3:
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
        elif self.ndim == 3:
            target_axis = [1, 2]
        else:
            raise ValueError("1D is not support")
        mean, var = tf.nn.moments(inputs, keep_dims=True, axes=target_axis)
        inputs = (inputs - mean) / k.sqrt(var + 1e-8) * self.gamma + self.beta
        return inputs


class Convbnrelu(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides, padding="same"):
        super(Convbnrelu, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                            kernel_size=kernel_size,
                                            strides=strides,
                                            padding=padding,
                                            use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization(momentum=.9)

        # self.bn = GLN()
    def call(self, inputs, training=None, mask=None):
        return tf.nn.relu(self.bn(self.conv(inputs), training))

class Auxgateoutput(tf.keras.Model):

    def __init__(self, input_channels, target_channels):
        super(Auxgateoutput, self).__init__()
        self.out = tf.keras.layers.Conv1D(input_channels, kernel_size=1)
        self.gate = tf.keras.layers.Conv1D(input_channels, kernel_size=1)
        self.maskconv1x1 = tf.keras.layers.Conv1D(target_channels, kernel_size=1, use_bias=False)

    def call(self, inputs, training=None, mask=None):
        gateoutput = tf.nn.tanh(self.out(inputs)) * tf.nn.sigmoid(self.gate(inputs))
        return self.maskconv1x1(gateoutput)


class Embedding_network(tf.keras.Model):
    def __init__(self, num_spks):
        super(Embedding_network, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=5
                                            , strides=1, padding="same", activation="relu")
        self.conv2 = Convbnrelu(filters=32, kernel_size=3, strides=(1, 2))
        self.conv3 = Convbnrelu(filters=64, kernel_size=3, strides=(1, 2))
        self.conv4 = Convbnrelu(filters=128, kernel_size=3, strides=(1, 2))
        self.conv5 = Convbnrelu(filters=256, kernel_size=3, strides=(1, 2))
        self.conv6 = Convbnrelu(filters=256, kernel_size=3, strides=(1, 2))
        self.conv7 = Convbnrelu(filters=512, kernel_size=3, strides=(1, 2))
        self.conv8 = Convbnrelu(filters=1024, kernel_size=3, strides=(1, 2))
        self.conv9 = tf.keras.layers.Conv2D(filters=256, kernel_size=1, strides=1, use_bias=False)
        self.face = CMFace(units=num_spks)
        self.pool = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, inputs, return_embedding=False, training=True, mask=None):
        if training:
            inputs, labels = inputs

        inputs = tf.abs(tf.signal.stft(inputs, frame_step=128, frame_length=256, fft_length=256))[:, :, 1:, tf.newaxis]
        inputs = self.conv1(tf.log1p(inputs))
        inputs = self.conv2(inputs, training)
        inputs = self.conv3(inputs, training)
        inputs = self.conv4(inputs, training)
        inputs = self.conv5(inputs, training)
        inputs = self.conv6(inputs, training)
        inputs = self.conv7(inputs, training)
        inputs = self.conv8(inputs, training)
        inputs = self.conv9(inputs)
        if not return_embedding:
            if training:
                inputs = self.pool(inputs)
                inputs = self.face([inputs, labels], training=True)
            else:
                inputs = self.pool(inputs)
                inputs = self.face(inputs, training=False)
            return inputs
        else:
            return tf.nn.l2_normalize(self.pool(inputs), axis=-1)



"""class Verification(tf.keras.Model):
    def __init__(self, num_spks):
        super(Verification, self).__init__()
        self.bilstm = tf.keras.layers.CuDNNGRU(units=512)

        self.dense0 = tf.keras.layers.Dense(units=512, use_bias=True, activation="relu")
        self.dense = tf.keras.layers.Dense(units=num_spks, use_bias=False)

    def call(self, inputs, training=None, mask=None):
        inputs = self.dense(self.dense0(self.bilstm(inputs[..., 0])))
        return inputs"""

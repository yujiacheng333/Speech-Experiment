import tensorflow as tf
from Classifier.arcface import Cosface


class ConvBn(tf.keras.Model):
    def __init__(self, filters, kernel_size=3, stride=(1, 1)):
        super(ConvBn, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                           strides=stride, padding="same", use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, mask=None):
        inputs = self.conv(inputs)
        inputs = self.bn(inputs, training)
        return inputs


class IdentityBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size=3):
        super(IdentityBlock, self).__init__()
        self.conv0 = ConvBn(filters=filters, kernel_size=kernel_size)
        self.conv1 = ConvBn(filters=filters, kernel_size=kernel_size)
        
    def call(self, inputs, training=None, mask=None):
        res = inputs
        inputs = self.conv0(inputs)
        inputs = self.conv1(tf.nn.relu(inputs))
        return tf.nn.relu(res + inputs)


class ConvBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size=3):
        super(ConvBlock, self).__init__()
        self.conv0 = ConvBn(filters=filters, kernel_size=kernel_size, stride=2)
        self.conv1 = ConvBn(filters=filters, kernel_size=kernel_size)
        self.skip = ConvBn(filters=filters, stride=2, kernel_size=1)

    def call(self, inputs, training=None, mask=None):
        res = self.skip(inputs, training)
        inputs = self.conv1(self.conv0(inputs, training), training)
        return res + inputs

"""class Resnet18(tf.keras.Model):
    def __init__(self, num_spk):
        super(Resnet18, self).__init__()
        self.convinput = tf.keras.layers.Conv2D(filters=64, strides=2, kernel_size=3)
        self.res2x_1 = BottleNeck(filters=64)
        self.res2x_2 = BottleNeck(filters=64)
        self.resize_2 = ConvBnRelu(filters=128)
        self.res3x_1 = BottleNeck(filters=128)
        self.res3x_2 = BottleNeck(filters=128)
        self.resize_3 = ConvBnRelu(filters=256)
        self.res4x_1 = BottleNeck(filters=256)
        self.res4x_2 = BottleNeck(filters=256)
        self.resize_4 = ConvBnRelu(filters=512)
        self.res5x_1 = BottleNeck(filters=512)
        self.res5x_2 = BottleNeck(filters=512)
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(num_spk, use_bias=False)
    def call(self, inputs, training=None, mask=None):
        inputs = self.convinput(inputs)
        inputs = self.res2x_1(inputs, training)
        inputs = self.res2x_2(inputs, training)
        inputs = self.resize_2(inputs, training)
        inputs = self.res3x_1(inputs, training)
        inputs = self.res3x_2(inputs, training)
        inputs = self.resize_3(inputs, training)
        inputs = self.res4x_1(inputs, training)
        inputs = self.res4x_2(inputs, training)
        inputs = self.resize_4(inputs, training)
        inputs = self.res5x_1(inputs, training)
        inputs = self.res5x_2(inputs, training)
        inputs = self.pool(inputs)
        inputs = self.dense(inputs)
        return inputs"""

class Resnet18(tf.keras.Model):
    def __init__(self, num_spk=256):
        super(Resnet18, self).__init__()
        self.convinput = tf.keras.layers.Conv2D(filters=64, strides=2, kernel_size=3)

        self.res2x_1 = IdentityBlock(filters=64)
        self.resize_2 = IdentityBlock(filters=64)
        self.resize_3 = ConvBlock(filters=128)
        self.res3x_1 = IdentityBlock(filters=128)
        self.resize_4 = ConvBlock(filters=256)
        self.res4x_1 = IdentityBlock(filters=256)
        self.res5x_1 = ConvBlock(filters=512)
        self.res5x_2 = IdentityBlock(filters=512)
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.face = Cosface(units=num_spk)
    def call(self, inputs, training=None, mask=None):
        if training:
            inputs, labels = inputs
        inputs = tf.abs(tf.signal.stft(inputs, frame_step=128, frame_length=256, fft_length=256))[:, :, 1:, tf.newaxis]
        inputs = self.convinput(inputs)
        inputs = self.res2x_1(inputs, training)
        inputs = self.resize_2(inputs, training)
        inputs = self.res3x_1(inputs, training)
        inputs = self.resize_3(inputs, training)
        inputs = self.res4x_1(inputs, training)
        inputs = self.resize_4(inputs, training)
        inputs = self.res5x_1(inputs, training)
        inputs = self.res5x_2(inputs, training)
        inputs = self.pool(inputs)
        if training:
            inputs = self.face([inputs, labels], training=True)
        else:
            inputs = self.face(inputs, training=False)
        return inputs
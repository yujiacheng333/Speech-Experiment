# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.keras import backend as k

class Encoder(tf.keras.Model):
    def __init__(self, kernel_size, out_channels):
        super(Encoder, self).__init__()
        self.conv = tf.keras.layers.Conv1D(kernel_size=kernel_size, filters=out_channels, use_bias=False,
                                           strides=kernel_size//2, padding="same")

    def call(self, inputs, training=None, mask=None):
        inputs = tf.nn.relu(self.conv(inputs))
        return inputs


class Decoder(tf.keras.Model):
    def __init__(self, kernel_size):
        super(Decoder, self).__init__()
        self.conv = tf.keras.layers.Conv1DTranspose(kernel_size=kernel_size, filters=1, use_bias=False,
                                                    strides=kernel_size//2, padding="same")

    def call(self, inputs, training=None, mask=None):
        return self.conv(inputs)


class ConvStem(tf.keras.Model):
    def __init__(self, hidden):
        super(ConvStem, self).__init__()
        self.forward = tf.keras.Sequential([tf.keras.layers.Conv1D(filters=hidden, kernel_size=1),
                                            tfa.layers.GroupNormalization(1),
                                            tf.keras.layers.PReLU(shared_axes=[1]),
                                            tf.keras.layers.Conv1D(filters=hidden, groups=hidden, kernel_size=5, strides=1, padding="same"),
                                            tfa.layers.GroupNormalization(1),
                                            tf.keras.layers.PReLU(shared_axes=[1])])

    def call(self, inputs, training=None, mask=None):
        return self.forward(inputs)


class ConvOut(tf.keras.Model):
    def __init__(self, outputp_channels):
        super(ConvOut, self).__init__()
        self.forwad = tf.keras.Sequential([tfa.layers.GroupNormalization(1),
                                           tf.keras.layers.PReLU(shared_axes=[1]),
                                           tf.keras.layers.Conv1D(filters=outputp_channels, kernel_size=1),
                                           tfa.layers.GroupNormalization(1)])
        self.activation = tf.keras.layers.PReLU(shared_axes=[1])

    def call(self, inputs, training=None, mask=None):
        inputs, res = inputs
        inputs = self.forwad(inputs)
        return self.activation(inputs+res)


class ConvSubSample(tf.keras.Model):
    def __init__(self, input_channels, kernel_size, stride=2):
        super(ConvSubSample, self).__init__()
        self.Convs = tf.keras.Sequential([tf.keras.layers.Conv1D(filters=input_channels,
                                                                 kernel_size=kernel_size,
                                                                 groups=input_channels,
                                                                 strides=stride, padding="same"),
                                          tfa.layers.GroupNormalization(1),
                                          tf.keras.layers.PReLU(shared_axes=[1])])

    def call(self, inputs, training=None, mask=None):
        inputs = self.Convs(inputs)
        return inputs


class Ublock(tf.keras.Model):
    def __init__(self, input_chs=128, u_hidden=512, sample_depth=4):
        super(Ublock, self).__init__()
        self.sample_depth = sample_depth
        self.stemlayer = ConvStem(hidden=u_hidden)
        self.Donwsamples = [ConvSubSample(input_channels=u_hidden,
                                          kernel_size=5,
                                          stride=2) for _ in range(sample_depth)]
        self.convoutput = ConvOut(outputp_channels=input_chs)

    def call(self, inputs, training=None, mask=None):
        outputs = [self.stemlayer(inputs)]
        for i in range(self.sample_depth):
            outputs.append(self.Donwsamples[i](outputs[-1]))
        temp = outputs.pop(-1)
        for i in range(self.sample_depth):
            temp = k.repeat_elements(temp, 2, axis=1) + outputs.pop(-1)
        temp = self.convoutput([temp, inputs])
        return temp


class Separator(tf.keras.Model):
    def __init__(self, enc_channels=512, hidden_channels=128, u_hidden=512, sample_depth=4, repeat=4):
        super(Separator, self).__init__()
        self.sample_depth = sample_depth
        self.forward = tf.keras.Sequential([tfa.layers.GroupNormalization(1),
                                            tf.keras.layers.Conv1D(filters=hidden_channels, kernel_size=1,
                                                                   use_bias=True)])
        for i in range(repeat):
            self.forward.add(Ublock(u_hidden=u_hidden, sample_depth=sample_depth))
        self.forward.add(tf.keras.layers.Conv1D(filters=enc_channels, kernel_size=1,
                                                use_bias=True, activation="relu"))
        self.pad_size = None

    def pad_data(self, x):
        if self.pad_size is None:
            assert x.shape[1] > 2 ** self.sample_depth, "tai duan le"
            self.pad_size = (2 ** self.sample_depth) - int(x.shape[1]) % (2 ** self.sample_depth)
        return tf.pad(x, ((0, 0), (self.pad_size, 0), (0, 0)))

    def call(self, inputs, training=None, mask=None):
        return self.forward(self.pad_data(inputs))[:, self.pad_size:]


class SEmodelM(tf.keras.Model):
    def __init__(self,
                 input_seq_len=16384,
                 kernel_size=21,
                 sample_depth=4,
                 enc_channels=512,
                 hidden_channels=128,
                 uhidden=512):
        super(SEmodelM, self).__init__()
        self.sample_depth = sample_depth
        self.kernel_size = kernel_size
        self.pad_size = (kernel_size//2)-input_seq_len % (kernel_size//2)
        self.pad = tf.keras.layers.ZeroPadding1D(padding=(0, self.pad_size))
        self.corp = tf.keras.layers.Cropping1D(cropping=(0, self.pad_size))
        self.stl = Encoder(kernel_size=kernel_size, out_channels=enc_channels)
        self.istl = Decoder(kernel_size=kernel_size)
        self.separator = Separator(enc_channels=enc_channels,
                                   hidden_channels=hidden_channels,
                                   sample_depth=self.sample_depth, u_hidden=uhidden)

    def call(self, inputs, training=None, mask=None):
        inputs = self.pad(inputs[..., None])
        e = self.stl(inputs)
        m = self.separator(e)
        inputs = self.istl(e*m)
        inputs = self.corp(inputs)
        return inputs[..., 0]


if __name__ == '__main__':
    test_inputs = tf.ones([1, 16384])
    mod = SEmodelM()
    mod(test_inputs)
    mod.separator.summary()

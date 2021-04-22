import tensorflow as tf
import tensorflow_addons as tfa


class Encoder(tf.keras.Model):
    def __init__(self, kernel_size, out_channels=256):
        super(Encoder, self).__init__()
        self.conv = tf.keras.layers.Conv1D(kernel_size=kernel_size, filters=out_channels, use_bias=False,
                                           strides=kernel_size//2, padding="same")

    def call(self, inputs, training=None, mask=None):
        inputs = tf.nn.relu(self.conv(inputs))
        return inputs


class Decoder(tf.keras.Model):
    def __init__(self, kernel_size=2):
        super(Decoder, self).__init__()
        self.conv = tf.keras.layers.Conv1DTranspose(kernel_size=kernel_size, filters=1, use_bias=False,
                                                    strides=kernel_size//2, padding="same")

    def call(self, inputs, training=None, mask=None):
        inputs = self.conv(inputs)
        return tf.squeeze(inputs)


class ConvTasNetBlock(tf.keras.Model):
    def __init__(self, kernel_size=3, inputs_chs=256, skip_chs=256, dilation_rate=1, expand_rate=2):
        super(ConvTasNetBlock, self).__init__()
        self.convlist = tf.keras.Sequential([tf.keras.layers.Conv1D(filters=inputs_chs*expand_rate,
                                                                    kernel_size=1),
                                             tf.keras.layers.PReLU(shared_axes=1),
                                             tfa.layers.GroupNormalization(1),
                                             tf.keras.layers.Conv1D(filters=inputs_chs*expand_rate,
                                                                    kernel_size=kernel_size,
                                                                    dilation_rate=dilation_rate,
                                                                    groups=inputs_chs*expand_rate,
                                                                    padding="same"),
                                             tf.keras.layers.PReLU(shared_axes=1),
                                             tfa.layers.GroupNormalization(1)])
        self.conv_res = tf.keras.layers.Conv1D(filters=inputs_chs, kernel_size=1)
        self.conv_skip = tf.keras.layers.Conv1D(filters=skip_chs, kernel_size=1)

    def call(self, inputs, training=None, mask=None):
        res = inputs
        inputs = self.convlist(inputs)
        return res + self.conv_res(inputs), self.conv_skip(inputs)


class ConvTasNetBranch(tf.keras.Model):
    def __init__(self,
                 enc_channels=512,
                 bottleneck_channels=128,
                 blocknum=8,
                 expand_ratio=4,
                 repeat=3):
        super(ConvTasNetBranch, self).__init__()
        self.conv_input = tf.keras.Sequential([tfa.layers.GroupNormalization(),
                                               tf.keras.layers.Conv1D(filters=bottleneck_channels,
                                                                      kernel_size=1)])
        self.blocks_list = []
        for i in range(repeat):
            for j in range(blocknum):
                self.blocks_list.append(ConvTasNetBlock(inputs_chs=bottleneck_channels,
                                                        skip_chs=bottleneck_channels,
                                                        dilation_rate=2**j,
                                                        expand_rate=expand_ratio))
        self.conv_output = tf.keras.Sequential([tf.keras.layers.PReLU(shared_axes=1),
                                                tf.keras.layers.Conv1D(filters=enc_channels,
                                                                       kernel_size=1,
                                                                       activation="sigmoid")])

    def call(self, inputs, training=None, mask=None):
        inputs = self.conv_input(inputs)
        skips = 0.
        for block in self.blocks_list:
            inputs, skip = block(inputs)
            skips = skip + skips
        inputs += skips
        inputs = self.conv_output(inputs)
        return inputs


class SEmodelM(tf.keras.Model):
    def __init__(self, length=16, enc_channels=512, bottleneck_channels=128, blocknum=8, expand_ratio=4, repeat=3):
        super(SEmodelM, self).__init__()
        self.stl = Encoder(kernel_size=length, out_channels=enc_channels)
        self.istl = Decoder(kernel_size=length)
        self.branch = ConvTasNetBranch(enc_channels=enc_channels,
                                       bottleneck_channels=bottleneck_channels,
                                       blocknum=blocknum,
                                       expand_ratio=expand_ratio,
                                       repeat=repeat)

    def call(self, inputs, training=None, mask=None):
        e = self.stl(inputs[..., None])
        m = self.branch(e)
        return self.istl(e*m)

if __name__ == '__main__':
    test_inputs = tf.ones([1, 16384])
    mode = SEmodelM()
    mode(test_inputs)
    mode.summary()

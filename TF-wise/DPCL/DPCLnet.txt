import tensorflow as tf
from tensorflow.python.keras import backend as k
eps = 1e-8


class GLN(tf.keras.Model):
    def __init__(self):
        super(GLN, self).__init__()
        self.ndim = None
        self.gamma = None
        self.beta = None

    def build(self, input_shape):
        chs = input_shape[-1]
        self.gamma = self.add_variable(name="gamma",
                                       shape=[1, 1, chs],
                                       initializer=tf.keras.initializers.Ones(),
                                       trainable=True)
        self.beta = self.add_variable(name="beta",
                                      shape=[1, 1, chs],
                                      initializer=tf.keras.initializers.Zeros(),
                                      trainable=True)

    def call(self, inputs, training=None, mask=None):
        mean, var = tf.nn.moments(inputs, keep_dims=True, axes=[1, 2])
        inputs = (inputs - mean) / k.sqrt(var + eps) * self.gamma + self.beta
        return inputs


class BiLSTM(tf.keras.Model):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.model = tf.keras.Sequential([tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNGRU(units=1024, return_sequences=True)),
                                          tf.keras.layers.Conv1D(filters=1024, kernel_size=1),
                                          GLN(), tf.keras.layers.PReLU()])

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)


class DPCLnet(tf.keras.Model):
    def __init__(self):
        super(DPCLnet, self).__init__()
        self.block_0 = BiLSTM()
        self.conv = tf.keras.layers.Conv1D(filters=129*40, kernel_size=1, use_bias=False)

    def call(self, inputs, training=None, mask=None):
        inputs = self.block_0(inputs)
        inputs = self.conv(inputs)
        inputs = tf.reshape(inputs, [-1, 247, 129, 40])
        return tf.nn.l2_normalize(inputs, axis=-1)


if __name__ == '__main__':
    tf.enable_eager_execution()
    test_input = tf.ones([64, 247, 129])
    mod = DPCLnet()
    test_input = mod(test_input)
    mod.summary()

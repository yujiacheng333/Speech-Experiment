import tensorflow as tf
from einops.layers.tensorflow import Rearrange
import tensorflow_addons as tfa
from tensorflow.keras.layers import (Dense, Dropout, LayerNormalization)
import numpy as np
eps = 1e-6


class FeedForward(tf.keras.Model):
    def __init__(self, hidden):
        super(FeedForward, self).__init__()
        self.ff = tf.keras.Sequential([tf.keras.layers.Dense(units=hidden * 4, activation=tfa.activations.gelu),
                                       tf.keras.layers.Dense(units=hidden)])
        self.norm = LayerNormalization()

    def call(self, inputs, training=None, mask=None):
        return self.norm(inputs + self.ff(inputs))
        
class TransformerBlock(tf.keras.Model):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super(TransformerBlock, self).__init__()
        self.att = tfa.layers.MultiHeadAttention(head_size=embed_dim // num_heads,
                                                 num_heads=num_heads,
                                                 output_size=embed_dim,
                                                 use_projection_bias=True,
                                                 return_attn_coef=False)
        # MultiHeadSelfAttention(embed_dim, num_heads)
        self.mlp = tf.keras.Sequential(
            [
             tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=mlp_dim, return_sequences=True)),
             tf.keras.layers.ReLU(),
             Dense(embed_dim)])
        self.norm0 = tf.keras.layers.LayerNormalization(axis=-1)
        self.norm1 = tf.keras.layers.LayerNormalization(axis=-1)

    def call(self, inputs, training=None, **kwargs):
        res = inputs
        inputs = self.att([inputs, inputs, inputs])
        inputs = self.norm0(inputs + res)
        res = inputs
        inputs = self.mlp(inputs)
        return self.norm1(inputs + res)
        

class Encoder(tf.keras.Model):

    def __init__(self, kernel_size, out_channels=256):
        super(Encoder, self).__init__()
        self.conv = tf.keras.layers.Conv1D(kernel_size=kernel_size, filters=out_channels, use_bias=False,
                                           strides=kernel_size//2, padding="same")

    def call(self, inputs, training=None, mask=None):
        """
        input_shape = [Bs, samplepts, chs~1 or not]
        """
        inputs = tf.nn.relu(self.conv(inputs))
        return inputs


class Decoder(tf.keras.Model):
    def __init__(self, kernel_size=2):
        super(Decoder, self).__init__()
        self.conv = tf.keras.layers.Conv1DTranspose(kernel_size=kernel_size, filters=1, use_bias=False,
                                                    strides=kernel_size//2, padding="same")

    def call(self, inputs, training=None, mask=None):
        """
        input_shape = [Bs, samplepts//stride, channels]
        output_shape = [Bs, samplepts, spk]
        """
        inputs = self.conv(inputs)
        return tf.squeeze(inputs)

def segmentation(x, k):
    bs, length, hidden = x.shape
    target_length = int(np.floor(length*2/k))
    gap = (target_length * k//2) - length
    x = tf.pad(x, ((0, 0), (k//2-k//4, gap+k//4), (0, 0)))
    x = Rearrange("b l f->(b f) l")(x)
    x = tf.signal.frame(x, frame_length=k, frame_step=k//2)
    x = Rearrange("(b a) t f -> b t f a", b=bs)(x)
    return x, gap

def over_add(x, pad_len):
    bs, n, k, hidden = x.shape
    x = Rearrange("b n k h->(b h) n k")(x)
    x = tf.signal.overlap_and_add(x, frame_step=k//2)
    x = x[:, k//2-k//4:-(pad_len+k//4)]
    x = Rearrange("(b h) l->b l h", h=hidden)(x)
    return x


# class DPRNNblock(tf.keras.Model):
#     def __init__(self,
#                  feature_channels,
#                  hidden_channels):
#         super(DPRNNblock, self).__init__()
#         self.intra_rnn = tf.keras.layers.Bidirectional(
#             tf.keras.layers.LSTM(units=hidden_channels, return_sequences=True))
#         self.inter_rnn = tf.keras.layers.Bidirectional(
#             tf.keras.layers.LSTM(units=hidden_channels, return_sequences=True))
#         self.intra_norm = tfa.layers.GroupNormalization()
#         self.inter_norm = tfa.layers.GroupNormalization()
#         self.intra_linear = tf.keras.layers.Conv1D(filters=feature_channels, kernel_size=1, use_bias=False)
#         self.inter_linear = tf.keras.layers.Conv1D(filters=feature_channels, kernel_size=1, use_bias=False)
#
#     def call(self, inputs, training=None, mask=None):
#         """
#         input_shape = [Bs, N, K, chs]
#         eq shape mapping
#         """
#         output = inputs
#         Bs, N, K, chs = inputs.get_shape().as_list()
#         inputs = tf.reshape(inputs, [Bs*N, K, chs])
#         inputs = self.intra_rnn(inputs)
#         inputs = self.intra_linear(inputs)
#         inputs = tf.reshape(inputs, [Bs, N, K, chs])
#         inputs = self.intra_norm(inputs)
#         output += inputs
#
#         inputs = tf.reshape(tf.transpose(output, [0, 2, 1, 3]), [Bs*K, N, chs])
#         inputs = self.inter_rnn(inputs)
#         inputs = self.inter_linear(inputs)
#         inputs = tf.transpose(tf.reshape(inputs, [Bs, K, N, chs]), [0, 2, 1, 3])
#         inputs = self.inter_norm(inputs)
#         output += inputs
#         return output

class DPRNNblock(tf.keras.Model):
    def __init__(self,
                 feature_channels,
                 hidden_channels):
        super(DPRNNblock, self).__init__()
        self.intra_former = TransformerBlock(embed_dim=feature_channels, num_heads=4, mlp_dim=hidden_channels)
        self.inter_former = TransformerBlock(embed_dim=feature_channels, num_heads=4, mlp_dim=hidden_channels)

    def call(self, inputs, training=None, mask=None):
        """
        input_shape = [Bs, N, K, chs]
        eq shape mapping
        """
        res = inputs
        Bs, N, K, chs = inputs.shape
        inputs = tf.reshape(inputs, [Bs*N, K, chs])
        inputs = self.intra_former(inputs)
        inputs = tf.reshape(inputs, [Bs, N, K, chs]) + res
        res = inputs
        inputs = tf.reshape(tf.transpose(inputs, [0, 2, 1, 3]), [Bs*K, N, chs])
        inputs = self.inter_former(inputs)
        inputs = tf.transpose(tf.reshape(inputs, [Bs, K, N, chs]), [0, 2, 1, 3])
        return inputs + res


class Auxgateoutput(tf.keras.Model):
    """
    Aux gate output for DPRNN
    https://github.com/ShiZiqiang/dual-path-RNNs-DPRNNs-based-speech-separation/blob/master/models.py
    """
    def __init__(self, input_channels, target_channels):
        super(Auxgateoutput, self).__init__()
        self.out = tf.keras.layers.Conv1D(input_channels, kernel_size=1)
        self.gate = tf.keras.layers.Conv1D(input_channels, kernel_size=1, kernel_initializer=tf.keras.initializers.Zeros(),
                                           bias_initializer=tf.keras.initializers.Constant(1.))
        self.maskconv1x1 = tf.keras.layers.Conv1D(target_channels, kernel_size=1, use_bias=True)

    def call(self, inputs, training=None, mask=None):
        inputs = self.maskconv1x1(inputs)
        return tf.nn.tanh(self.out(inputs)) * tf.nn.sigmoid(self.gate(inputs))


class DPRNN(tf.keras.Model):
    """
    basic_model of DPRNN
    """
    def __init__(self, feature_channels, hidden_channels,
                 num_layers=6,
                 K=200):
        super(DPRNN, self).__init__()
        self.K = K
        self.norm = tfa.layers.GroupNormalization()
        self.dual_rnn = tf.keras.Sequential([])
        for i in range(num_layers):
            self.dual_rnn.add(DPRNNblock(hidden_channels=hidden_channels, feature_channels=feature_channels))
        self.dual_rnn.add(tf.keras.layers.PReLU(shared_axes=[1, 2, 3]))
        self.dual_rnn.add(tf.keras.layers.Dense(units=feature_channels))

    def call(self, inputs, training=None, mask=None):
        inputs, pad_len = segmentation(inputs, self.K)
        inputs = self.norm(inputs)
        # [Bs, N, K, chs]
        inputs = self.dual_rnn(inputs)
        inputs = over_add(inputs, pad_len)
        return inputs


class Model(tf.keras.Model):
    def __init__(self,
                 enc_channels,
                 feature_channels,
                 hidden_channels,
                 kernel_size=2,
                 num_layers=6,
                 K=200):
        super(Model, self).__init__()
        self.enc = Encoder(kernel_size=kernel_size, out_channels=enc_channels)
        self.dec = Decoder(kernel_size=kernel_size)
        self.sepration = DPRNN(feature_channels,
                               hidden_channels,
                               num_layers,
                               K)
        self.compressgate = Auxgateoutput(input_channels=enc_channels, target_channels=enc_channels)

    def call(self, inputs, training=None, mask=None):
        e = self.enc(inputs[..., tf.newaxis])
        s = self.sepration(e)
        s = tf.nn.relu(self.compressgate(s))
        audios = self.dec(s*e)
        return audios


if __name__ == '__main__':
    test_inputs = np.random.normal(size=[2, 16384])
    test_inputs = tf.cast(test_inputs, tf.float32)
    Model_basic = Model(enc_channels=64, kernel_size=2, feature_channels=64, hidden_channels=128, K=180)
    data = Model_basic(test_inputs)
    print(data.shape)
    Model_basic.summary()

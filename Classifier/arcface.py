import tensorflow as tf
from tensorflow import math


class Cosface(tf.keras.Model):
    def __init__(self, units, s=30., m=0.3):
        super(Cosface, self).__init__()
        self.w = None
        self.units = units
        self.s = s
        self.m = m
        self.eps = 1e-6

    def build(self, input_shape):
            self.w = self.add_variable(name='embedding_weights', shape=[input_shape[0][-1].value, self.units],
                                       dtype=tf.float32, trainable=True)

    def call(self, inputs, training=None, mask=None):

        if training:
            featurerep, labels = inputs
            labels = tf.cast(tf.one_hot(labels, depth=self.units), tf.float32)
            embedding = tf.nn.l2_normalize(featurerep, axis=-1)
            weights = tf.nn.l2_normalize(self.w, axis=0)
            cos_t = tf.matmul(embedding, weights)
            cos_t_m = cos_t - self.m
            cos_t_m = tf.clip_by_value(cos_t_m, clip_value_min=-1., clip_value_max=1.)
            inv_labels = 1. - labels
            output = self.s * (cos_t * inv_labels + cos_t_m * labels)
        else:
            output = tf.matmul(inputs, self.w)
        return output


class CMFace(tf.keras.Model):
    def __init__(self, units, s=30., m1=1.1, m2=0.5, m3=0.4):
        super(CMFace, self).__init__()
        self.w = None
        self.units = units
        self.s = s
        self.m_list = [m1, m2, m3]
        self.eps = 1e-6

    def build(self, input_shape):
        try:
            self.w = self.add_variable(name='embedding_weights', shape=[input_shape[0][-1].value, self.units],
                                       dtype=tf.float32, trainable=True)
        except:
            self.w = self.add_variable(name='embedding_weights', shape=[input_shape[-1].value, self.units],
                                       dtype=tf.float32, trainable=True)

    def call(self, inputs, training=None, mask=None):

        if training:
            featurerep, labels = inputs
            labels = tf.cast(tf.one_hot(labels, depth=self.units), tf.float32)
            embedding = tf.nn.l2_normalize(featurerep, axis=-1)
            weights = tf.nn.l2_normalize(self.w, axis=0)
            cos_t = tf.matmul(embedding, weights)
            # cos_t = tf.clip_by_value(cos_t, clip_value_min=-1., clip_value_max=1.)
            theta = tf.math.acos(cos_t)
            theta_m = theta * self.m_list[0] + self.m_list[1]
            theta_m = tf.where(theta_m < 3.1415926535, theta, theta_m)
            cos_t_m = tf.math.cos(theta_m) - self.m_list[2]
            cos_t_m = tf.where(cos_t_m > -1., cos_t_m, cos_t)  # release margin
            inv_labels = 1. - labels
            output = self.s * (cos_t * inv_labels + cos_t_m * labels)
        else:
            output = tf.matmul(inputs, self.w)
        return output
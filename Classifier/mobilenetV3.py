import tensorflow as tf

def hardsigmoid(x):
    return tf.nn.relu6(x + 3.) / 6.

def hardswitch(x):
    return x * hardsigmoid(x)


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


class Conv1x1Bn(tf.keras.Model):
    def __init__(self, filters, stride=1, kernel_size=1):
        super(Conv1x1Bn, self).__init__()
        self.ptconv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, use_bias=False, strides=stride,
                                             padding="same")
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.9)

    def call(self, inputs, training=None, mask=None):
        inputs = self.ptconv(inputs)
        inputs = self.bn(inputs, training)
        return inputs


class BottleNeck(tf.keras.Model):
    def __init__(self, input_filters, target_filters, expand_filters, kernel_size, activation, se, stride):
        super(BottleNeck, self).__init__()
        self.stride = stride
        self.activation = activation
        self.se = se
        if self.se:
            self.seblock = SeModel(input_chs=expand_filters, reduction=4)
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.9)
        self.conv1 = Conv1x1Bn(filters=expand_filters)
        self.convd = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size,
                                                     use_bias=False, strides=stride, padding="same")
        self.conv2 = Conv1x1Bn(filters=target_filters)
        if input_filters != target_filters and stride == 1:
            self.skip = True
            self.skip_ = Conv1x1Bn(filters=target_filters, stride=1)
        else:
            self.skip = False

    def call(self, inputs, training=None, mask=None):
        res = inputs
        inputs = self.activation(self.conv1(inputs, training))
        inputs = self.convd(inputs)
        inputs = self.activation(self.bn(inputs, training))
        if self.se:
            inputs = self.seblock(inputs, training)
        inputs = self.conv2(inputs, training)
        if self.skip:
            res = self.skip_(res, training)
            return inputs + res
        elif self.stride == 1:
            return inputs + res

        else:
            return inputs


class MobilenetV3(tf.keras.Model):
    def __init__(self, num_classes):
        super(MobilenetV3, self).__init__()

        self.conv1 = Conv1x1Bn(filters=16, kernel_size=3, stride=2)
        """BottleNecks"""
        self.conv2 = Conv1x1Bn(filters=576)
        self.pool = tf.keras.layers.GlobalMaxPool2D()
        self.dense0 = tf.keras.layers.Dense(units=1280, use_bias=False)
        self.dense1 = tf.keras.layers.Dense(units=num_classes, use_bias=False)

        # cascade
        self.bot0 = BottleNeck(kernel_size=3, input_filters=16, expand_filters=16, target_filters=16, activation=tf.nn.relu,
                                se=True, stride=2)
        self.bot1 = BottleNeck(kernel_size=3, input_filters=16, expand_filters=72, target_filters=24, activation=tf.nn.relu,
                               se=False, stride=2)
        self.bot2 = BottleNeck(kernel_size=3, input_filters=24, expand_filters=88, target_filters=24, activation=tf.nn.relu,
                               se=False, stride=1)
        self.bot3 = BottleNeck(kernel_size=5, input_filters=24, expand_filters=96, target_filters=40, activation=hardswitch,
                               se=True, stride=2)
        self.bot4 = BottleNeck(kernel_size=5, input_filters=40, expand_filters=240, target_filters=40, activation=hardswitch,
                               se=True, stride=1)
        self.bot5 = BottleNeck(kernel_size=5, input_filters=40, expand_filters=240, target_filters=40, activation=hardswitch,
                               se=True, stride=1)
        self.bot6 = BottleNeck(kernel_size=5, input_filters=40, expand_filters=120, target_filters=48, activation=hardswitch,
                               se=True, stride=1)
        self.bot7 = BottleNeck(kernel_size=5, input_filters=48, expand_filters=144, target_filters=48, activation=hardswitch,
                               se=True, stride=1)
        self.bot8 = BottleNeck(kernel_size=5, input_filters=48, expand_filters=288, target_filters=96, activation=hardswitch,
                               se=True, stride=2)
        self.bot9 = BottleNeck(kernel_size=5, input_filters=96, expand_filters=576, target_filters=96, activation=hardswitch,
                               se=True, stride=1)
        self.bot10 = BottleNeck(kernel_size=5, input_filters=96, expand_filters=576, target_filters=96, activation=hardswitch,
                               se=True, stride=1)

    def call(self, inputs, training=None, mask=None):
        if training:
            inputs, labels = inputs
        inputs = tf.abs(tf.signal.stft(inputs, frame_step=128, frame_length=256, fft_length=256))[:, :, 1:, tf.newaxis]
        inputs = tf.math.log1p(inputs)
        inputs = hardswitch(self.conv1(inputs, training))
        inputs = self.bot0(inputs, training)
        inputs = self.bot1(inputs, training)
        inputs = self.bot2(inputs, training)
        inputs = self.bot3(inputs, training)
        inputs = self.bot4(inputs, training)
        inputs = self.bot5(inputs, training)
        inputs = self.bot6(inputs, training)
        inputs = self.bot7(inputs, training)
        inputs = self.bot8(inputs, training)
        inputs = self.bot9(inputs, training)
        inputs = self.bot10(inputs, training)
        inputs = hardswitch(self.conv2(inputs, training))
        inputs = self.pool(inputs)
        inputs = hardswitch(self.dense0(inputs))
        inputs = self.dense1(inputs)
        return inputs
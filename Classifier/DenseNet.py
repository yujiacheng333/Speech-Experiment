import tensorflow as tf

class DenseConvUnit(tf.keras.Model):
    def __init__(self, filters, kernel_size):
        super(DenseConvUnit, self).__init__()
        self.conv1x1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, padding="same")
        self.conv3x3 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding="same")
        self.norm0 = tf.keras.layers.BatchNormalization()
        self.norm1 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, mask=None):
        inputs = tf.nn.relu(self.norm0(inputs, training))
        inputs = self.conv1x1(inputs)
        inputs = tf.nn.relu(self.norm1(inputs, training))
        inputs = self.conv3x3(inputs)
        return inputs


class DenseBlock(tf.keras.Model):
    def __init__(self, k=32, stack_times=6):
        super(DenseBlock, self).__init__()
        self.modellist = []
        for i in range(stack_times):
            self.modellist.append(DenseConvUnit(filters=k, kernel_size=3))

    def call(self, inputs, training=None, mask=None):
        to_next = [inputs]
        for layer in self.modellist:
            to_next.append(layer(tf.concat(to_next, axis=-1), training))
        return tf.concat(to_next, axis=-1)


class Transitionlayer(tf.keras.Model):
    def __init__(self, hidden):
        super(Transitionlayer, self).__init__()
        self.norm = tf.keras.layers.BatchNormalization()
        self.conv = tf.keras.layers.Dense(units=hidden)

    def call(self, inputs, training=None, mask=None):
        inputs = self.norm(inputs, training)
        inputs = tf.nn.relu(inputs)
        inputs = self.conv(inputs)
        return inputs


class DenseNetBackBone(tf.keras.Model):
    def __init__(self, stacklist=None, stem_hidden=64, k=32):
        super(DenseNetBackBone, self).__init__()
        if stacklist is None:
            stacklist = [6, 12, 24, 16]
        self.stacklist = stacklist
        self.convstem = tf.keras.layers.Conv2D(filters=stem_hidden,
                                               kernel_size=7,
                                               strides=2,
                                               padding="same")
        self.stempool = tf.keras.layers.MaxPool2D()
        self.block_list = []
        self.pool_list = []
        self.transition_list = []
        ini_trans = stem_hidden
        for i, stack in enumerate(stacklist):
            self.block_list.append(DenseBlock(k=k, stack_times=stack))
            if i != len(stacklist)-1:
                self.pool_list.append(tf.keras.layers.AveragePooling2D())
                ini_trans = self.transition_caculator(ini_trans, k, stack)//2
                self.transition_list.append(Transitionlayer(hidden=ini_trans))

    @staticmethod
    def transition_caculator(k0, k, repeat_times):
        return k*repeat_times + k0

    def call(self, inputs, training=None, mask=None):
        inputs = self.convstem(inputs)
        inputs = self.stempool(inputs)
        for i in range(len(self.stacklist)-1):
            inputs = self.block_list[i](inputs, training)
            inputs = self.transition_list[i](inputs, training)
            inputs = self.pool_list[i](inputs)
        inputs = self.block_list[-1](inputs)
        return inputs


class DenseNet121(tf.keras.Model):
    def __init__(self, category):
        super(DenseNet121, self).__init__()
        self.BackBone = DenseNetBackBone()
        self.pool = tf.keras.layers.GlobalMaxPool2D()
        self.decision = tf.keras.layers.Dense(units=category)

    def call(self, inputs, training=None, mask=None):
        embedding = self.pool(self.BackBone(inputs))
        score = self.decision(embedding)
        return score, embedding


if __name__ == '__main__':
    test_input = tf.ones([32, 224, 224, 3])
    model = DenseNet121(category=1000)
    model(test_input)
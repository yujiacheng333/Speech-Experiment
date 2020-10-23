class ResNet34(tf.keras.Model):
    def __init__(self, category):
        super(ResNet34, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.pool = tf.keras.layers.MaxPool2D()
        self.bneck1 = StrideBlock(filters=64, kernel_size=3, stride=1)
        self.bneck2 = Identityblock(filters=64, kernel_size=3)
        self.bneck3 = Identityblock(filters=64, kernel_size=3)

        self.bneck4 = StrideBlock(filters=128, kernel_size=3, stride=2)
        self.bneck5 = Identityblock(filters=128, kernel_size=3)
        self.bneck6 = Identityblock(filters=128, kernel_size=3)
        self.bneck7 = Identityblock(filters=128, kernel_size=3)

        self.bneck8 = StrideBlock(filters=256, kernel_size=3, stride=2)
        self.bneck9 = Identityblock(filters=256, kernel_size=3)
        self.bneck10 = Identityblock(filters=256, kernel_size=3)
        self.bneck11 = Identityblock(filters=256, kernel_size=3)
        self.bneck12 = Identityblock(filters=256, kernel_size=3)
        self.bneck13 = Identityblock(filters=256, kernel_size=3)

        self.bneck14 = StrideBlock(filters=512, kernel_size=3, stride=2)
        self.bneck15 = Identityblock(filters=512, kernel_size=3)
        self.bneck16 = Identityblock(filters=512, kernel_size=3)
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.decision = tf.keras.layers.Dense(units=category)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.pool(x)
        x = self.bneck1(x, training=training)
        x = self.bneck2(x, training=training)
        x = self.bneck3(x, training=training)
        x = self.bneck4(x, training=training)
        x = self.bneck5(x, training=training)
        x = self.bneck6(x, training=training)
        x = self.bneck7(x, training=training)
        res = x
        x = self.bneck8(x, training=training)
        x = self.bneck9(x, training=training)
        x = self.bneck10(x, training=training)
        x = self.bneck11(x, training=training)
        x = self.bneck12(x, training=training)
        x = self.bneck13(x, training=training)
        x = self.bneck14(x, training=training)
        x = self.bneck15(x, training=training)
        x = self.bneck16(x, training=training)
        x = self.decision(self.pool(x))
        return x


if __name__ == '__main__':
    model = ResNet34(category=1000)
    model.build(input_shape=(None, 256, 256, 3))
    model.summary()
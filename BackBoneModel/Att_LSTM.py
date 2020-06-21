import tensorflow as tf


class ATTLSTMcell(tf.keras.layers.Layer):
    def __init__(self, hidden, menmory):
        super(ATTLSTMcell, self).__init__()
        """
        input_shape = outputshape = hidden_size
        """
        self.menmory = menmory
        self.embedding_menmory = None
        self.hidden_size = hidden
        self.state_size = [hidden, hidden, hidden]
        self.denseI = tf.keras.layers.Dense(units=hidden)
        self.denseF = tf.keras.layers.Dense(units=hidden)
        self.denseC = tf.keras.layers.Dense(units=hidden)
        self.denseO = tf.keras.layers.Dense(units=hidden)
        self.recurent_activation = tf.keras.activations.hard_sigmoid
        self.activation = tf.keras.activations.tanh

    def call(self, inputs, states, training=None):
        h_tm1, r_tm1, c_tm1 = states
        pack = tf.concat([h_tm1, r_tm1, inputs], axis=-1)
        i_t = self.recurent_activation(self.denseI(pack))
        f_t = self.recurent_activation(self.denseF(pack))
        c_t = self.activation(self.denseC(pack))
        o_t = self.recurent_activation(self.denseO(pack))
        c_t = f_t * c_tm1 + i_t * c_t
        h_t = o_t * c_t
        h_t += inputs
        attention_weight = tf.nn.softmax(tf.einsum("bf,nf->bn", h_t, self.menmory), axis=-1)
        r_t = tf.einsum("bn,nf->bf", attention_weight, self.menmory)
        return h_t, [h_t, r_t, c_t]

if __name__ == '__main__':
    tf.enable_eager_execution()
    step = 5
    embedding = tf.zeros([3, 50])
    input = tf.broadcast_to(embedding[:, tf.newaxis], [3, step, 50])

    lstm = tf.keras.layers.RNN(cell=ATTLSTM_cell(hidden=50, menmory=tf.ones([50, 50])), return_sequences=False)
    lstm(input)

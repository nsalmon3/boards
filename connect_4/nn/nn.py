import tensorflow.keras as tf

from connect_4.implementations import *
 
class connect_4_nn(tf.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Inputs
        self.board_input = tf.layers.Input(shape = (6, 7), name = "board")
        self.id_input = tf.layers.Input(shape = (7, ), name = 'id')

        # Layers for board convolution embedding
        self.convolutional_1 = tf.layers.Conv2D(64, (3, 3), padding = "same")
        self.convolutional_2 = tf.layers.Conv2d(32, (3, 3), padding = "same")

        # Layers for id embedding
        self.dense_1 = tf.layers.Dense(16)
        self.dense_2 = tf.layers.Dense(16)

        # Layers after concatenation
        self.dense_3 = tf.layers.Dense(16)
        self.dense_output = tf.layers.Dense(7 + 4, activation = 'softmax')
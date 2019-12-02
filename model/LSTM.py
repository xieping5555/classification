from __future__ import print_function, division
from __future__ import absolute_import
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.python.ops.rnn import dynamic_rnn as rnn

from model.embLayer import Embeddings


class LSTM(Embeddings):
    def __init__(self):
        super().__init__()
        super().embedding_layer()

    def model(self):
        # (Bi-)RNN layers(-s)
        with tf.name_scope('GRU_layer'):
            rnn_outputs, _ = rnn(LSTMCell(self.hidden_size), inputs=self.batch_embedded, dtype=tf.float32)
            tf.summary.histogram('RNN_outputs', rnn_outputs)
            rnn_outputs = tf.reduce_mean(rnn_outputs, axis=2)
            print('rnn_outpts.shape:', rnn_outputs.shape)
            self.output = tf.reduce_sum(rnn_outputs, axis=1)


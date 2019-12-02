from __future__ import print_function, division
from __future__ import absolute_import
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn

from model.embLayer import Embeddings

from model.attention import Attention


class BiGRU_Attention(Embeddings):
    def __init__(self):
        super().__init__()
        super().embedding_layer()

    def model(self):
        # (Bi-GRU) layers
        rnn_outputs, _ = bi_rnn(GRUCell(self.hidden_size), GRUCell(self.hidden_size), inputs=self.batch_embedded,
                                dtype=tf.float32)
        tf.summary.histogram('RNN_outputs', rnn_outputs)

        if isinstance(rnn_outputs,tuple):
            rnn_outputs = tf.concat(rnn_outputs, 2)
            print('rnn_outputs.shape:', rnn_outputs.shape)
            rnn_outputs = tf.reduce_mean(rnn_outputs, axis=2)
            print('rnn_outputs.shape:',rnn_outputs.shape)
            self.output = tf.reduce_sum(rnn_outputs,axis=1)


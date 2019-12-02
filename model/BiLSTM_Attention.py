from __future__ import print_function, division
from __future__ import absolute_import
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn

from model.embLayer import Embeddings

from model.attention import Attention


class BiGRU_Attention(Embeddings):
    def __init__(self):
        super().__init__()
        super().embedding_layer()

    def model(self):
        # (Bi-GRU) layers
        rnn_outputs, _ = bi_rnn(LSTMCell(self.hidden_size), LSTMCell(self.hidden_size), inputs=self.batch_embedded,
                                dtype=tf.float32)
        tf.summary.histogram('RNN_outputs', rnn_outputs)

        # Attention layers
        with tf.name_scope('Attention_layer'):
            attention_ = Attention(rnn_outputs, self.attention_size, time_major=False, return_alphas=True)
            self.attention_output, alphas = attention_.attentionModel()
            tf.summary.histogram('alphas', alphas)
            print('attention_output.shape:', self.attention_output.shape)


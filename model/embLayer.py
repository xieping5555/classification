from __future__ import print_function,division
from __future__ import absolute_import

import tensorflow as tf
from model.paramers import Papramers

class Embeddings(Papramers):
    def __init__(self):
        super().__init__() # 调用父类中的参数
        
    def embedding_layer(self):
        # Different placeholder
        with tf.name_scope('Input'):
            self.batch_ph = tf.placeholder(tf.int32, [None, self.sequence_length], name='batch_ph')
            self.target_ph = tf.placeholder(tf.float32, [None], name='target_ph')
            self.seq_len_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')  # 实际的句子长度
            self.keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob_ph')

        # Embedding layer
        with tf.name_scope('Embedding_layer'):
            self.embeddings_var = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_dim], -1.0, 1.0,seed=self.seed),trainable=True)
            tf.summary.histogram('embeddings_var', self.embeddings_var)
            self.batch_embedded = tf.nn.embedding_lookup(self.embeddings_var, self.batch_ph)
            print('self.batch_embedded:', self.batch_embedded.shape)

      
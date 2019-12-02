from __future__ import print_function,division
from __future__ import absolute_import
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
from attention import Attention
from layer import FFN


class train(object):
    def __init__(self,X_train,y_train,X_test,y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.sequence_length = 150 # 序列长度 or 句子最大长度
        self.embedding_dim = 64 # model_dim
        self.model_dim = 64
        self.num_head = 1 # self-attention
        self.linear_key_dim = 64
        self.linear_value_dim = 64
        self.dropout = 0.2
        self.batch_size = 100
        self.model_path = './savemodel/train'
        self.numb_epoch = 10
        self.num_layers = 1
        self.vocab_size = 100
        self.delta = 0.5
        self.keep_prob = 0.8

    def positional_encoding(self,dim, sentence_length, dtype=tf.float32):
        """
        dim: 可自定义
        sentenc_length:句子长度"""

        encoded_vec = np.array([pos / np.power(10000, 2 * i / dim) for pos in range(sentence_length) for i in range(dim)])
        encoded_vec[::2] = np.sin(encoded_vec[::2])
        encoded_vec[1::2] = np.cos(encoded_vec[1::2])

        return tf.convert_to_tensor(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)

    def define_layer(self):
        # Different placeholders
        with tf.name_scope('Input'):
            self.batch_ph = tf.placeholder(tf.int32,[None,self.sequence_length],name='batch_ph')
            self.target_ph = tf.placeholder(tf.float32,[None],name='target_ph')
            self.keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob_ph')
            
        # Embedding layer
        with tf.name_scope('Embedding_layer'):
            self.embeddings_var = tf.Variable(tf.random_uniform([self.vocab_size,self.embedding_dim],-1.0,1.0),trainable=True)
            tf.summary.histogram('embeddings_var',self.embeddings_var)
            self.batch_embedded = tf.nn.embedding_lookup(self.embeddings_var,self.batch_ph)
            print('self.batch_embedded:',self.batch_embedded.shape) # shape [?,self.sequence_length,self.embedding_dim]
            
        # Positional encoding
        with tf.name_scope('positional_encoding'):
            positional_encoded = self.positional_encoding(self.model_dim,
                                                    self.sequence_length,
                                                    dtype = tf.float32)

            position_embedded = tf.nn.embedding_lookup(positional_encoded,self.batch_ph)
            print('position_embedded.shape:',position_embedded.shape)
            
            
            encoded_inputs = tf.add(self.batch_embedded,position_embedded)
        
        encoder_emb_inp = tf.nn.dropout(encoded_inputs,1.0-self.dropout)
        
        # self-attention
        with tf.name_scope('self-attention'):
            o1 = tf.identity(encoder_emb_inp)
            attention_ = Attention(num_heads=self.num_head,
                                  masked=False,
                                  linear_key_dim=self.linear_key_dim,
                                  linear_value_dim=self.vocab_size,
                                  model_dim=self.model_dim,
                                  dropout = self.dropout)
            multi_head_output = attention_.multi_head(o1,o1,o1)

        o2 = tf.contrib.layers.layer_norm(tf.add(o1, multi_head_output))
        # ffn = FFN(w1_dim=self.model_dim, w2_dim=self.model_dim, dropout=self.dropout)
        o1 = tf.identity(o2)

        # multi-layers
        # with tf.name_scope('multi-layers'):
        #     for i in range(1,self.num_layers+1):
        #         with tf.variable_scope(f'layer-{i}'):
        #             o2 = tf.contrib.layers.layer_norm(tf.add(o1,multi_head_output))
        #             ffn = FFN(w1_dim=self.model_dim,w2_dim=self.model_dim,dropout=self.dropout)
        #             o21 = ffn.dense_relu_dense(o2)
        #             o3 = tf.contrib.layers.layer_norm(tf.add(o2,o21))
        #             o1 = tf.identity(o3)

        o4 = tf.reduce_sum(o1, axis=2)
        print('o1.shape:',o4.shape)
        # add_layer
        with tf.name_scope('fully_layer'):
            inputsize = int(o4.shape[-1])
            w = tf.Variable(tf.random_normal([inputsize,1],-0.05,0.05))
            f = tf.matmul(o4,w)
            y_hat = tf.squeeze(f)
        
        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=self.target_ph))
            tf.summary.scalar('loss', loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-3,
                                               beta1 = 0.9,
                                               beta2 = 0.999,
                                               epsilon = 1e-08,
                                               use_locking = True).minimize(loss)

            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.sigmoid(y_hat)), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', accuracy)

        y_hat_ = tf.round(tf.sigmoid(y_hat))
        merged = tf.summary.merge_all() # 合并summary operation，运行初始化变量
        # Batch genetators
        train_batch_generator = self.batch_generator(self.X_train, self.y_train, self.batch_size)
        test_batch_generator = self.batch_generator(self.X_test, self.y_test, self.batch_size)

        session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        saver = tf.train.Saver()

        train_writer = tf.summary.FileWriter('./logdir/train', accuracy.graph)
        test_writer = tf.summary.FileWriter('./logdir/test', accuracy.graph)

        with tf.Session(config=session_conf) as sess:
            sess.run(tf.global_variables_initializer())
            print('Start learning...')
            fp = open('./result.txt', 'w', encoding='utf8')
            for epoch in range(self.numb_epoch):
                fp.write(str(epoch) + '\n')
                print('epoch:%d' % epoch)
                loss_train = 0
                loss_test = 0
                accuracy_train = 0
                accuracy_test = 0

                print('epoch:{}\t'.format(epoch), end="")
                # Training
                num_batches = self.X_train.shape[0] // self.batch_size
                train_yT, train_pred, alphas_qq, alphas_qq1 = [], [], [], []
                for b in tqdm(range(num_batches)):
                    x_batch, y_batch, indice = next(train_batch_generator)
                    loss_tr, acc, _, summary, y_pred = sess.run([loss, accuracy, optimizer, merged, y_hat_],
                                                                feed_dict={self.batch_ph: x_batch,
                                                                           self.target_ph: y_batch,
                                                                           self.keep_prob_ph: self.keep_prob})

                    train_yT.extend(y_batch.tolist())
                    train_pred.extend(y_pred.tolist())
                    accuracy_train += acc
                    loss_train = loss_tr * self.delta + loss_train * (1 - self.delta)
                    train_writer.add_summary(summary, b + num_batches * epoch)
                accuracy_train /= num_batches
                precision_train = metrics.precision_score(train_yT, train_pred, average='macro')
                recall_train = metrics.recall_score(train_yT, train_pred)
                f1_train = metrics.f1_score(train_yT, train_pred)
                print('loss:{:.4f},acc:{:.4f},precision:{:.4f},recall:{:.4f},f1_score:{:.4f}'.format(
                    loss_train, accuracy_train, precision_train, recall_train, f1_train))
                fp.write('train_loss:' + str(loss_train) + ' ' + 'train_acc:' + str(
                    accuracy_train) + ' ' + 'train_precision:' + str(precision_train) +
                         ' ' + 'train_recall:' + str(recall_train) + ' ' + 'train_f1_score:' + str(f1_train) + '\n')

                # Testing
                test_yT, test_pred = [], []
                num_batches = self.X_test.shape[0] // self.batch_size
                for b in tqdm(range(num_batches)):
                    x_batch, y_batch, indice = next(test_batch_generator)
                    loss_test_batch, acc, summary, y_pred = sess.run([loss, accuracy, merged, y_hat_],
                                                                     feed_dict={self.batch_ph: x_batch,
                                                                                self.target_ph: y_batch,
                                                                                self.keep_prob_ph: 1.0})

                    test_yT.extend(y_batch.tolist())
                    test_pred.extend(y_pred.tolist())
                    accuracy_test += acc
                    loss_test += loss_test_batch
                    test_writer.add_summary(summary, b + num_batches * epoch)
                accuracy_test /= num_batches
                loss_test /= num_batches
                precision_test = metrics.precision_score(test_yT, test_pred)
                recall_test = metrics.recall_score(test_yT, test_pred)
                f1_test = metrics.f1_score(test_yT, test_pred)
                print(
                    'loss_test:{:.4f},accuracy_test:{:.4f},precision_test:{:.4f},recall_test:{:.4f},f1_score_test:{:.4f}'.format(
                        loss_test, accuracy_test, precision_test, recall_test, f1_test))
                fp.write('test_loss:' + str(loss_test) + ' ' + 'test_acc:' + str(
                    accuracy_test) + ' ' + 'test_precision:' + str(precision_test) +
                         ' ' + 'test_recall:' + str(recall_test) + ' ' + 'test_f1_score:' + str(f1_test) + '\n')
                saver.save(sess, self.model_path + str(epoch))

            train_writer.close()
            test_writer.close()
            fp.close()
            print("Run 'tensorboard --logdir=./logdir' to checkout tensorboard logs.")

    def batch_generator(self, X, y, batch_size):
        """Primitive batch generator"""
        size = X.shape[0]
        X_copy = X.copy()
        y_copy = y.copy()
        indices = np.arange(size)
        np.random.shuffle(indices)
        X_copy = X_copy[indices]
        y_copy = y_copy[indices]
        i = 0
        while True:
            if i + batch_size <= size:
                yield X_copy[i:i + batch_size], y_copy[i:i + batch_size], indices[i:i + batch_size]
                i += batch_size
            else:
                i = 0
                indices = np.arange(size)
                np.random.shuffle(indices)
                X_copy = X_copy[indices]
                y_copy = y_copy[indices]
                continue

def getdataset():
    import dill
    # 数据读取与查看
    path = 'C:/Users/lejon/Desktop/phishing classfication/VonPhishingData_all/data/fengData/'
    dill_file = path + "vonDataset20180426.dill"
    with open(dill_file, 'rb') as f:
        pickleData = dill.load(f)
        train_x, train_y = pickleData["train_x"], pickleData["train_y"]
        test_x, test_y = pickleData["test_x"], pickleData["test_y"]

    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    import time
    t1 = time.time()
    X_train, y_train, X_test, y_test = getdataset()
    t2 = time.time()
    print(str(t2 - t1))
    train_ = train(X_train, y_train, X_test, y_test)
    train_.define_layer()
    print(time.time() - t2)
        
        
            
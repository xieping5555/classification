
import numpy as np
import tensorflow as tf

from attention import positional_encoding
from encoder import Encoder

class Model:
    def __init__(self,
                 X_train,
                 y_train,
                 X_test,
                 y_test,
                 vocab_size=100,
                 model_dim=50,
                 max_seq_length=150,
                 dropout=0.2,
                 num_layers=2,
                 num_heads=2,
                 linear_value_dim=60,
                 ffn_dim=50,
                 target_vocab_size=1,
                 dtype=tf.float32):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.dtype = dtype
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.max_seq_length = max_seq_length
        self.dropout = dropout
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.linear_value_dim = linear_value_dim
        self.ffn_dim = ffn_dim
        self.batch_size = 256 
        self.target_vocab_size = target_vocab_size # 可理解为分类数
        self.target_ph = tf.placeholder(dtype,[None],name='target_ph')
        self.model_path = './savemodel/transforer'
    
    def train(self):
        with tf.name_scope('Input'):
            self.batch_ph = tf.placeholder(tf.int32, [None, self.max_seq_length], name='batch_ph')
            self.target_ph = tf.placeholder(tf.float32, [None], name='target_ph')
        # Embedding layers
        with tf.name_scope('Embedding_layers'):
            # Word Embedding
            # embedding_encoder = tf.get_variable([self.vocab_size,self.model_dim])
            embedding_encoder = tf.Variable(tf.random_uniform([self.vocab_size,self.model_dim], -1.0, 1.0),trainable=True)
          
            # Position Encoding
        with tf.variable_scope("positional-encoding"):
            positional_encoded = positional_encoding(self.model_dim,
                                                     self.max_seq_length,
                                                     dtype=self.dtype)

            # Add
            position_inputs = tf.tile(tf.range(0,self.max_seq_length),[self.batch_size]) # tf.tile # 复制tensor，对tf.range()以self.batch_size进行复制
            position_inputs = tf.reshape(position_inputs,[self.batch_size,self.max_seq_length]) # batch_size x [0,1,2,...,n]
            
            embedding_inputs = embedding_encoder
               
            encoded_inputs = tf.add(tf.nn.embedding_lookup(embedding_inputs,self.batch_ph),
                                tf.nn.embedding_lookup(positional_encoded,position_inputs))    
        
        encoder_emb_inp = tf.nn.dropout(encoded_inputs,1.0-self.dropout)
        
        with tf.variable_scope("Encoder"):
            encoder = Encoder(num_layers=self.num_layers,
                              num_heads=self.num_heads,
                              linear_value_dim=self.linear_value_dim,
                              model_dim = self.model_dim,
                              ffn_dim = self.ffn_dim)
            encoder_outputs = encoder.build(encoder_emb_inp)  
            
        with tf.variable_scope("Output"):
            outputs = tf.layers.dense(encoder_outputs, self.target_vocab_size)
        
        # Cross-entropy loss and optimizer initizlization
        with tf.name_scope('Metrics'):
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=outputs, labels=self.target_ph))
            tf.summary.scalar('loss', loss)
            
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
            
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.sigmoid(outputs)), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', accuracy)
           
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
            saver.restore(sess, self.model_path)
            fp = open('./result.txt', 'w', encoding='utf8')
            for epoch in range(11,self.numb_epoch):
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
                    loss_tr, acc, y_pred = sess.run([loss, accuracy, self.outputs],
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
                    loss_test_batch, acc,y_pred = sess.run([loss, accuracy,self.outputs],
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
    
    train_ = Model(X_train, y_train, X_test, y_test)
    train_.train()
    print(time.time() - t2)

        
            
                    
            
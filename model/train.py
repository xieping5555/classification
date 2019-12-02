from __future__ import print_function,division
from __future__ import absolute_import
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn import metrics
from model.BiGRU_Attention import BiGRU_Attention

class Train(BiGRU_Attention):
    def __init__(self):
        super().__init__()
        super().model()

    def batch_generator(self, X, y, batch_size):
        """Primitive batch generator"""
        size = X.shape[0]
        print('size:'+str(size))
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
 
    def train(self):
        # Batch genetators
        self.output = tf.reduce_sum(self.attention_output, axis=1)
        self.y_hat = tf.squeeze(self.output)

        # optimizer
        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_hat, labels=self.target_ph))
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.loss)

            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.sigmoid(self.y_hat)), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.y_hat_ = tf.round(tf.sigmoid(self.y_hat))
        self.merged = tf.summary.merge_all()  # 合并summary operation，运行初始化变量

        train_batch_generator = self.batch_generator(self.X_train, self.y_train, self.batch_size)
        test_batch_generator = self.batch_generator(self.X_test, self.y_test, self.batch_size)

        session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        saver = tf.train.Saver()

        train_writer = tf.summary.FileWriter(self.model_path+'/logdir/train', self.accuracy.graph)
        test_writer = tf.summary.FileWriter(self.model_path+'./logdir/test', self.accuracy.graph)
        
        with tf.Session(config=session_conf) as sess:
            sess.run(tf.global_variables_initializer())
            print('Start learning...')
            fp = open(self.model_path+'/result.txt', 'w', encoding='utf8')
            for epoch in range(17,self.numb_epoch):
                if epoch!=0:
                    saver.restore(sess, self.model_path)
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
                    loss_tr, acc, _, summary, y_pred = sess.run([self.loss, self.accuracy, self.optimizer, self.merged, self.y_hat_],
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
                fp.write('train_loss:' + str(loss_train) + ' ' + 'train_acc:' + str(accuracy_train) + ' ' + 'train_precision:' + str(precision_train) +
                         ' ' + 'train_recall:' + str(recall_train) + ' ' + 'train_f1_score:' + str(f1_train) + '\n')

                saver.save(sess, self.model_path)

                # Testing
                test_yT, test_pred = [], []
                num_batches = self.X_test.shape[0] // self.batch_size
                for b in tqdm(range(num_batches)):
                    x_batch, y_batch, indice = next(test_batch_generator)
                    loss_test_batch, acc, summary, y_pred = sess.run([self.loss, self.accuracy, self.merged, self.y_hat_],
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
                print('loss_test:{:.4f},accuracy_test:{:.4f},precision_test:{:.4f},recall_test:{:.4f},f1_score_test:{:.4f}'.format(
                        loss_test, accuracy_test, precision_test, recall_test, f1_test))
                fp.write('test_loss:' + str(loss_test) + ' ' + 'test_acc:' + str(accuracy_test) + ' ' + 'test_precision:' + str(precision_test) +
                         ' ' + 'test_recall:' + str(recall_test) + ' ' + 'test_f1_score:' + str(f1_test) + '\n')

            train_writer.close()
            test_writer.close()
            fp.close()
            print("Run 'tensorboard --logdir=./logdir' to checkout tensorboard logs.")

if __name__=='__main__':
    Train().train()

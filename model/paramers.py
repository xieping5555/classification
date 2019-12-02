from model import inputData

class Papramers():
    def __init__(self):
        self.X_train,self.y_train,self.X_test,self.y_test,self.X_val,self.y_val = inputData.getdataset() # 获取数据
        self.sequence_length = 150
        self.embedding_dim = 128 # 就是降维
        self.hidden_size = 60
        self.hidden_layer = 120
        self.attention_size = 80  # 此处设置的原因尚不清楚
        self.batch_size = 256  # 一轮输入的样本量
        self.numb_epoch = 20  # 迭代的轮次
        self.keep_prob = 0.8
        self.delta = 0.5
        self.vocab_size = 100  # 词汇量大小
        self.model_path = 'C:/Users/lejon/Desktop/phishing classfication/VonPhishingData_all/model3/saveModel/BiGRUAttention/Bigru_att'
        self.seed = 317
        
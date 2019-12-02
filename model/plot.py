import matplotlib.pyplot as plt

# 数据读入
def lstm():
    path = 'C:/Users/lejon/Desktop/phishing classfication/VonPhishingData_all/model3/BILSTM/em_lstm/'
    data = open(path+'result_lstm.txt').readlines()
    lstm_att_train = data[1:21]
    lstm_att_test = data[23:43]
    lstm_att_train = [float(each.split('acc:')[1].split(',')[0]) for each in lstm_att_train]
    lstm_att_test = [float(each.split('accuracy_test:')[1].split(',')[0]) for each in lstm_att_test]
    return lstm_att_train,lstm_att_test

def att_lstm():
    path = 'C:/Users/lejon/Desktop/phishing classfication/VonPhishingData_all/model3/att_BiGRU/att41Lstm/'
    data = open(path+'result_att4.txt').readlines()
    lstm_att_train = data[1:21]
    lstm_att_test = data[23:43]
    lstm_att_train = [float(each.split('acc:')[1].split(',')[0]) for each in lstm_att_train]
    lstm_att_test = [float(each.split('accuracy_test:')[1].split(',')[0]) for each in lstm_att_test]
    return lstm_att_train,lstm_att_test

def gru():
    path = 'C:/Users/lejon/Desktop/phishing classfication/VonPhishingData_all/model3/BIGRU/GRU(128)/'
    data = open(path+'result_gru.txt').readlines()
    lstm_att_train = data[1:21]
    lstm_att_test = data[23:43]
    lstm_att_train = [float(each.split('acc:')[1].split(',')[0]) for each in lstm_att_train]
    lstm_att_test = [float(each.split('accuracy_test:')[1].split(',')[0]) for each in lstm_att_test]
    return lstm_att_train,lstm_att_test

def att_gru():
    path = 'C:/Users/lejon/Desktop/phishing classfication/VonPhishingData_all/model3/att_BiGRU/att6Gru/'
    data = open(path+'result_att51.txt').readlines()
    lstm_att_train = data[1:21]
    lstm_att_test = data[23:43]
    lstm_att_train = [float(each.split('loss:')[1].split(',')[0]) for each in lstm_att_train]
    lstm_att_test = [float(each.split('loss_test:')[1].split(',')[0]) for each in lstm_att_test]
    return lstm_att_train,lstm_att_test

def Bilstm():
    path = 'C:/Users/lejon/Desktop/phishing classfication/VonPhishingData_all/model3/BILSTM/BiLSTM/'
    data = open(path+'result_bilstm.txt').readlines()
    lstm_att_train = data[1:21]
    lstm_att_test = data[23:43]
    lstm_att_train = [float(each.split('acc:')[1].split(',')[0]) for each in lstm_att_train]
    lstm_att_test = [float(each.split('accuracy_test:')[1].split(',')[0]) for each in lstm_att_test]
    return lstm_att_train,lstm_att_test

def att_Bilstm():
    path = 'C:/Users/lejon/Desktop/phishing classfication/VonPhishingData_all/model3/att_BiGRU/att7LSTM/'
    data = open(path+'result_att7.txt').readlines()
    lstm_att_train = data[1:21]
    lstm_att_test = data[23:43]
    lstm_att_train = [float(each.split('acc:')[1].split(',')[0]) for each in lstm_att_train]
    lstm_att_test = [float(each.split('accuracy_test:')[1].split(',')[0]) for each in lstm_att_test]
    return lstm_att_train,lstm_att_test

def Bigru():
    path = 'C:/Users/lejon/Desktop/phishing classfication/VonPhishingData_all/model3/BIGRU/BiGRU(128)/'
    data = open(path+'result_bigru.txt').readlines()
    lstm_att_train = data[1:21]
    lstm_att_test = data[23:43]
    lstm_att_train = [float(each.split('acc:')[1].split(',')[0]) for each in lstm_att_train]
    lstm_att_test = [float(each.split('accuracy_test:')[1].split(',')[0]) for each in lstm_att_test]
    return lstm_att_train,lstm_att_test

def att_Bigru():
    path = 'C:/Users/lejon/Desktop/phishing classfication/VonPhishingData_all/model3/att_BiGRU/att6Gru/'
    data = open(path+'result.txt').readlines()
    lstm_att_train = data[1:21]
    lstm_att_test = data[23:43]
    lstm_att_train = [float(each.split('acc:')[1].split(',')[0]) for each in lstm_att_train]
    lstm_att_test = [float(each.split('accuracy_test:')[1].split(',')[0]) for each in lstm_att_test]
    return lstm_att_train,lstm_att_test

def att_bigru_fully():
    path = 'C:/Users/lejon/Desktop/phishing classfication/VonPhishingData_all/model4/BIGRU_embedding80/'
    with open(path+'result.txt',encoding='utf-8') as f:
        data = f.readlines()
    # data = open(path+'result.txt','rb').readlines()
    lstm_att_train = data[1:21]
    lstm_att_test = data[23:43]
    lstm_att_train = [float(each.split('acc:')[1].split(',')[0]) for each in lstm_att_train]
    lstm_att_test = [float(each.split('accuracy_test:')[1].split(',')[0]) for each in lstm_att_test]
    return lstm_att_train, lstm_att_test


# 数据集大小不同时候的情况
def plot_loss(x,y,y1,y2,name):
    plt.plot(x, y, marker='*', mec='r', ms=6, label=u'lstm loss')
    names = list(x)
    plt.plot(x, y1, marker='*', ms=6, label=u'Bigru loss')
    plt.plot(x, y2, marker='*', ms=6, label=u'Bigru_attition loss')
    plt.xlim((0, 16))
    plt.ylim(0, 0.09)
    plt.legend()
    plt.xticks(x, names, rotation=0)
    plt.margins(0)
    plt.subplots_adjust(bottom=0.1)
    plt.xlabel(u"epoch")  # X轴标签
    plt.ylabel("Loss")  # Y轴标签
    plt.title(name)
    plt.savefig("./"+name+'.png')
    plt.show()

def plot_acc(x,y0,y1,y2,y3,y4,y5,name):
    # plt.grid()
    plt.plot(x, y0, marker='*', mec='r', ms=6, label=u'LSTM')
    names = list(x)
    plt.plot(x, y1, marker='*', ms=6, label=u'GRU')
    plt.plot(x, y2, marker='*', ms=6, label=u'BiGRU',)
    # plt.text(15, 0.98, 'put some text')
    plt.plot(x, y3, marker='*', ms=6, label=u'BiLSTM-Attention')
    plt.plot(x, y4, marker='*', ms=6, label=u'BiGRU-Attention')
    plt.plot(x, y5, marker='*', ms=6, label=u'BiGRU-Attention-fully')
    plt.xlim((1, 21))
    plt.ylim(0.96, 1)
    plt.legend(loc='lower right') #['best','upper right','upper left','lower right','right','center left',
    # 'center right','lower center','upper center','center']
    plt.xticks(x, names, rotation=0)
    # plt.margins(0)
    plt.subplots_adjust(bottom=0.2)
    plt.xlabel(u"epoch")  # X轴标签
    plt.ylabel("accuracy")  # Y轴标签
    # plt.title(name)
    plt.savefig("./"+name+'.png')
    plt.show()
# plot_loss(x,lstm_train_loss,lstm_test_loss,'LSTM loss')
# plot_acc(x,lstm_train_acc,lstm_test_acc,'LSTM Accuracy')
# plot_loss(x,gru_train_loss,gru_test_loss,'GRU loss')
# plot_acc(x,gru_train_acc,gru_test_acc,'GRU Accuracy')
# plot_loss(x,att7_train_loss,att7_test_loss,'Attention Loss')
# plot_acc(x,att7_train_acc,att7_test_acc,'Attention Accuracy')

def plot_acc2(x,y1,y2,y3,y4,name):
    # plt.grid()
    plt.bar([1,5,9,13,17,21,25,29],y1,label='accuracy')
    plt.bar([2,6,10,14,18,22,26,30],y2,label='precision')
    plt.bar([3,7,11,15,19,23,27,31],y3,label='recall')
    plt.bar([4,8,12,16,20,24,28,32],y4,label='f1-score')
    plt.legend()
    plt.ylim(0.75, 1)
    plt.xlabel('Experiment')
    plt.ylabel('ratio%')
    plt.title(name)
    plt.savefig("./" + name + '.png')
    plt.show()


if __name__=="__main__":
    x = range(1, 21)
    lstm_train,lstm_test = lstm()
    # lstm_att_train,lstm_att_test = att_lstm()
    # plot_acc(x,lstm_train,lstm_test,lstm_att_train,lstm_att_test,"lstm and lstm+attention accuracy")

    gru_train, gru_test = gru()
    # gru_att_train,gru_att_test = att_gru()
    # plot_acc(x, gru_train, gru_test, gru_att_train, gru_att_test, "gru and gru+attention accuracy")

    # Bilstm_train, Bilstm_test = Bilstm()
    Bilstm_att_train, Bilstm_att_test = att_Bilstm()
    # plot_acc(x, Bilstm_train, Bilstm_test, Bilstm_att_train, Bilstm_att_test, "Bilstm and Bilstm+attention accuracy")

    Bigru_train, Bigru_test = Bigru()
    Bigru_att_train, Bigru_att_test = att_Bigru()
    bigru_att_fully_train,bigru_att_fully_test = att_bigru_fully()
    # plot_acc(x, lstm_train,lstm_test, Bigru_att_train, Bigru_att_test, "The accuracy of models in Train and Validation")
    plot_acc(x,lstm_test, gru_test,Bigru_test, Bilstm_att_test,Bigru_att_test,bigru_att_fully_test,"The accuracy of models in Train and Validation")
    # plot_acc(x, lstm_train, lstm_test, Bigru_att_train, Bigru_att_test,
    #          "The accuracy of BiGRU-Attention in Training and Validation")
    # plot_loss(x, lstm_test, Bigru_test, Bigru_att_test, "The accuracy of Train")

    # plot_acc(x,lstm_test,Bigru_test,Bigru_att_test, Bilstm_att_train, "The accuracy of Val")
    # plot_loss(x,lstm_test,Bigru_test,Bigru_att_test, "The accuracy of Train")





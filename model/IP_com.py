import dill
import pandas as pd

# 数据集重复样本情况
def duplicated_data():
    path = 'C:/Users/lejon/Desktop/papereading/phishing classfication/VonPhishingData_all/data/'
    train = pd.read_csv(path+'fengData/trainUrl.csv')['url']
    test = pd.read_csv(path+'fengData/valiUrl.csv')['url']

    path = 'C:/Users/lejon/Desktop/papereading/phishing classfication/VonPhishingData_all/data/'
    dill_file = path + "vonDataset20180426.dill"

    with open(dill_file, 'rb') as f:
        pickleData = dill.load(f)
        val_x, val_y = pickleData["test_x"], pickleData["test_y"]
        char_to_int = pickleData["char_to_int"]

    int_to_char = dict()
    for each in char_to_int:
        int_to_char[char_to_int[each]] = each
    val = []
    for each in val_x:
        each = [each0 for each0 in each if each0 > 0]
        a = pd.DataFrame()
        a['A'] = each
        each0 = a['A'].map(int_to_char).tolist()
        val.append(''.join(each0))


    train.drop_duplicates(keep='first', inplace=True)
    train.shape[0]

    a = pd.concat([train, test], axis=0)
    a.shape[0]
    # test12 phish
    testdata = pd.read_csv(path + 'PhishTank/verified2.csv')
    testdata.shape[0]

    # openphishi
    test3 = pd.read_csv(path + 'OpenPhishing/openphish.txt')
    test4 = pd.read_csv(path + 'whiteUrl.csv')

def TankData(char_to_int):
    # 数据读取与查看
    path = 'C:/Users/lejon/Desktop/papereading/phishing classfication/VonPhishingData_all/data/fengData/'
    TankUrl = pd.read_csv(path + 'valiUrl.csv')
    val_cramel = TankUrl[TankUrl.label == 0]

    testdata = val_cramel['url'].values
    test_x = []
    for each in testdata:
        a = pd.DataFrame()
        a['A'] = list(each)[:150]
        each0 = a['A'].map(char_to_int).fillna(0.0).tolist()
        test_x.append(np.array(each0))
    test_x = [np.array([0] * (150 - len(x)) + x.tolist()) for x in test_x]
    test_y = val_cramel['label']
    # [:70120] 比例1：10，[:35060] 比例1:5，[:7012] 比例1:1;6%:所有白
    return test_x[:7000], test_y[:7000]

# 数据域名统计
stat_ip = []
train_0 = train[train.label==0]['url'].values
train_1 = train[train.label==1]['url'].values
train_stat_ip0 = [each.split('://')[1].split('/')[0] for each in train_0]
train_stat_ip1 = [each.split('://')[1].split('/')[0] for each in train_1]
train_IP0 = list(set(train_stat_ip0)) #  训练数据集白样本Ip
train_IP1 = list(set(train_stat_ip1)) # 训练数据集黑样本Ip
test1 = [] # phishtank（2018.5-2019.3）
test1_ip1 = list(set(test1))
len(test1_ip1)
test3 = [] # openphish





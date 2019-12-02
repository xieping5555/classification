import pandas as pd
import dill

def getdataset():
    # 数据读取与查看
    path = 'C:/Users/lejon/Desktop/phishing classfication/VonPhishingData_all/data/fengData/'
    dill_file = path + "vonDataset20180426.dill"
    with open(dill_file, 'rb') as f:
        pickleData = dill.load(f)
        train_x, train_y = pickleData["train_x"], pickleData["train_y"]
        val_x, val_y = pickleData["val_x"], pickleData["val_y"]
        test_x, test_y = pickleData["test_x"], pickleData["test_y"]


    # train_x, train_y = duplicated(train_x, train_y)
    # val_x, val_y = duplicated(val_x, val_y)
    return train_x, train_y, test_x, test_y,val_x,val_y

def duplicated(data, label):
    data = data.T
    aa = pd.DataFrame()
    col_ = []
    for i in range(150):
        each = 'col_' + str(i)
        col_.append(each)
        aa[each] = data[i]
    aa['label'] = label
    aa.drop_duplicates(subset=col_, keep='first', inplace=True)
    data = aa[col_].values
    label = aa['label'].values
    return data, label
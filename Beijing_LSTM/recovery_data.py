#coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from LSTM_Model import split_dataset,load_dataset
from sklearn.preprocessing import LabelEncoder

def data_recovery(data,dataset1):
    dataset = pd.read_csv(data, header=0, index_col=0)
    values = dataset.values
    # integer encode direction
    encoder = LabelEncoder()
    values[:, 4] = encoder.fit_transform(values[:, 4])
    # ensure all data is float
    values = values.astype('float32')
    print("=======", values)
    Max=max(values[:,0])
    Min=min(values[:,0])
    print(Min,Max)
    result=np.loadtxt(dataset1)
    for i in range(len(result)):
        result[i]=result[i]*(Max-Min)+Min
    np.savetxt("result_1.txt",result)
    print(result)
    return result

data_recovery("pollution.csv","train_result.txt")

















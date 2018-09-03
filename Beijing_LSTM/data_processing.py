"""-*- coding: utf-8 -*-
 DateTime   : 2018/9/2 16:03
 Author  : Peter_Bonnie
 FileName    : Beijing_LSTM
 Software: PyCharm
"""
import pandas as pd
from datetime import datetime

"""
"""

#数据的预处理
def parse(x):
    return datetime.strptime(x,'%Y %m %d %H')

dataset=pd.read_csv("Beijing.csv",parse_dates=[['year','month','day','hour']],index_col=0,date_parser=parse)
dataset.drop('No',axis=1,inplace=True)
dataset.columns=['pollution','dew','temp','press','wnd_dir','wnd_spd','snow','rain']
dataset.index.name='date'
dataset['pollution'].fillna(0,inplace=True)
dataset=dataset[24:]
print(dataset.head(5))
dataset.to_csv('pollution.csv')  #存到另外一个csv文件中去



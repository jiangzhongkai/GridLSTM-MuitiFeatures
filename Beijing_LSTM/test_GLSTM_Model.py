"""-*- coding: utf-8 -*-
 DateTime   : 2018/10/9 11:46
 Author  : Peter_Bonnie
 FileName    : test_GLSTM_Model.py
 Software: PyCharm
"""
"""
利用测试集对已经训练的模型进行预测
"""
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math
import os
from tensorflow.contrib.rnn import BasicLSTMCell,LSTMCell,GridLSTMCell
from tensorflow.contrib.grid_rnn import Grid2LSTMCell,Grid2BasicLSTMCell
import time
from GridLSTM_model import *

#测试集的加载
test_dataset=pd.read_csv("test_dataset.csv",low_memory=False)
test_dataset_values=test_dataset['value'].values
print(test_dataset_values)
# print(test_dataset[test_dataset['value'].values==0].count())
# print(test_dataset_values)
timestep=1
test_dataset_X=[]
test_dataset_Y=[]
for i in range(len(test_dataset_values)-timestep):
    x=test_dataset_values[i:i+timestep]
    test_dataset_X.append(x)
    test_dataset_Y.append(test_dataset_values[i+timestep])

test_dataset_X=np.array(test_dataset_X)
test_dataset_Y=np.array(test_dataset_Y)
test_dataset_X=np.reshape(test_dataset_X,newshape=[-1,timestep,test_dataset_X.shape[1]])

test_dataset_Y=np.reshape(test_dataset_Y,newshape=[-1,1])

config=Config()
X=tf.placeholder(dtype=tf.float32,shape=[None,config.timesteps,config.features])
Y=tf.placeholder(dtype=tf.float32,shape=[None,config.outputdims])

epoch = config.training_epoch
lr = config.learning_rate
batch_size = config.batch_size
prediction_Y, _ = GridLSTM_Model(X, config)
cost = tf.reduce_mean(tf.square(tf.subtract(prediction_Y, Y)))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

#加载模型
sess=tf.Session()
saver=tf.train.Saver()
saver.restore(sess,"./Model/GLSTM_Model.ckpt")
#测试集用于验证模型的性能的
for i in range(100):
    for start, end in zip(range(0, len(test_dataset_X), batch_size), range(batch_size, len(test_dataset_X) + 1, batch_size)):
        loss_test,_,test_result = sess.run([cost,optimizer,prediction_Y], feed_dict={X: test_dataset_X, Y: test_dataset_Y})
        print("epoch{}====loss_test:{}".format(str(i),loss_test/len(batch_size)))   #计算平均的误差
        np.savetxt("test_result_7.txt",test_result)
sess.close()






















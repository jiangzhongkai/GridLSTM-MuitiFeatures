"""-*- coding: utf-8 -*-
 DateTime   : 2018/9/2 16:31
 Author  : Peter_Bonnie
 FileName    : LSTM_Model
 Software: PyCharm
"""
#进行多特征得预测和
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'  #当没有相应的GPU设备时，会使用CPU来运行。


#当步长为1的时候的情况:
#当步长为2的时候的情况：
# .....
#做数据集的分配，
#step_1:对数据集进行预处理并将其转化为有监督学习，对数据进行归一化处理
def series_to_supervised(data,n_in=1,n_out=1,dropnan=True):
    n_vars=1 if type(data) is list else data.shape[1]
    df=pd.DataFrame(data)
    cols,names=list(),list()

    for i in range(n_in,0,-1):
        cols.append(df.shift(i))
        names+=[('var%d(t-%d)' %(j+1,i)) for j in range(n_vars)]

    for i in range(0,n_out):
        cols.append(df.shift(-i))

        if i==0:
            names+=[('var%d(t)'%(j+1)) for j in range(n_vars)]
        else:
            names+=[('var%d(t+%d)'%(j+1,i)) for j in range(n_vars)]

    agg=pd.concat(cols,axis=1)
    agg.columns=names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def max_Min(dataset):
    """
    对数据进行归一化处理
    :param dataset:
    :return:
    """
    d1=dataset.shape[0]
    d2=dataset.shape[1]
    for i in range(d2):
        Max=max(dataset[:,i])
        Min=min(dataset[:,i])
        for j in range(d1):
            dataset[j,i]=(dataset[j,i]-Min)/(Max-Min)
    print(dataset)
    return dataset


def load_dataset(data):
    """
    加载数据集
    :param data:
    :return:
    """
    dataset = pd.read_csv(data, header=0, index_col=0)
    values = dataset.values
    # integer encode direction
    encoder = LabelEncoder()     #标准化标签,将标签格式转换为range()范围内的数
    values[:,4] = encoder.fit_transform(values[:,4])
    # ensure all data is float
    values = values.astype('float32')
    print("=======",values)
    # normalize features
    # scaler = MinMaxScaler(feature_range=(0, 1))
    scaled=max_Min(values)
    # scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)
    print(reframed)
    # drop columns we don't want to predict
    reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
    # print(reframed)
    return reframed

def split_dataset(data):
    """
    分割数据集
    :param data:
    :return:
    """
    reframed=load_dataset(data)
    values=reframed.values
    n_train_hours=365*24
    train=values[:n_train_hours,:]
    test=values[n_train_hours:,:]

    #split into in and out
    train_x,train_y=train[:,:-1],train[:,-1]
    test_x,test_y=test[:,:-1],test[:,-1]
    #reshape input to be 3D  [samples,timesteps,features]
    train_x=train_x.reshape(train_x.shape[0],1,train_x.shape[1])
    test_x=test_x.reshape(test_x.shape[0],1,test_x.shape[1])
    return np.array(train_x),np.array(train_y),np.array(test_x),np.array(test_y)

class Config():
    """
    配置类
    """
    def __init__(self,train_x,train_y):
        self.timesteps=1
        self.features=8
        self.outputdims=1
        self.batch_size=50

        self.learning_rate=0.0001
        self.training_epoch=100
        self.keep_prob=0.9
        self.hidden_nums=9
        self.hidden_two=18
        self.hidden_three=27

        self.W = {
            'hidden': tf.Variable(tf.random_normal([self.features, self.hidden_nums])),
            'output': tf.Variable(tf.random_normal([self.hidden_two, 1]))
        }
        self.biases = {
            'hidden': tf.Variable(tf.random_normal([self.hidden_nums])),
            'output': tf.Variable(tf.random_normal([1]))
        }

def LSTM_Model(input_data,config):
    """
    LSTM模型
    :param input_data:
    :param config:
    :return:
    """
    input_data=tf.transpose(input_data,[1,0,2])
    input_data=tf.reshape(input_data,[-1,config.features])
    input_data=tf.nn.sigmoid(tf.matmul(input_data,config.W['hidden'])+config.biases['hidden'])
    input_data=tf.split(input_data,config.timesteps,0)
    #lsem_cell
    lstm_cell1=tf.nn.rnn_cell.BasicLSTMCell(num_units=config.hidden_nums,forget_bias=1.0,state_is_tuple=True)
    lstm_cell1=tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell1,output_keep_prob=config.keep_prob)
    lstm_cell11=tf.nn.rnn_cell.BasicLSTMCell(num_units=config.hidden_two,forget_bias=1.0,state_is_tuple=True)
    lstm_cell11=tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell11,output_keep_prob=config.keep_prob)
    lstm_cell2=tf.nn.rnn_cell.BasicLSTMCell(num_units=config.hidden_two,forget_bias=1.0,state_is_tuple=True)
    lstm_cell2=tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell2,output_keep_prob=config.keep_prob)
    stack_lstm=tf.nn.rnn_cell.MultiRNNCell(cells=[lstm_cell1,lstm_cell11,lstm_cell2],state_is_tuple=True)
    init_state=stack_lstm.zero_state(batch_size=config.batch_size,dtype=tf.float32)
    # print(type(input_data))
    outputs,_=tf.nn.static_rnn(cell=stack_lstm,inputs=input_data,dtype=tf.float32)
    output=tf.matmul(outputs[-1],config.W['output'])+config.biases['output']

    return output


def main():
    train_x, train_y, test_x, test_y = split_dataset('pollution.csv')
    print(train_x.shape)
    train_y = train_y.reshape([-1, 1])
    test_y = test_y.reshape([-1, 1])
    print(train_y.shape)

    # 定义一些占位符与变量
    config = Config(train_x, train_y)
    X = tf.placeholder(dtype=tf.float32, shape=[None, config.timesteps, config.features])
    Y = tf.placeholder(dtype=tf.float32, shape=[None, config.outputdims])
    epoch = config.training_epoch
    lr = config.learning_rate
    batch_size = config.batch_size

    prediction_Y = LSTM_Model(X, config)

    #利用MSE来做度量标准
    cost=tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(prediction_Y,Y)))))
    #cost = tf.reduce_mean(tf.square(prediction_Y - Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

    # 生成saver
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    test_losses = []
    train_losses = []

    for i in range(epoch):
        train_total_loss = 0.0
        test_total_loss = 0.0
        # print("============epoch:",str(i+1),"======================")
        for start, end in zip(range(0, len(train_x), batch_size), range(batch_size, len(train_x) + 1, batch_size)):
            sess.run(optimizer, feed_dict={X: train_x[start:end], Y: train_y[start:end]})
            loss_train, train_result = sess.run([cost, prediction_Y], feed_dict={X: train_x, Y: train_y})
            loss_test, test_result = sess.run([cost, prediction_Y], feed_dict={X: test_x, Y: test_y})
            test_total_loss += loss_test
            train_total_loss += loss_train
        test_losses.append(test_total_loss)
        train_losses.append(train_total_loss)
        print("epoch:{}======loss_train:{}====loss_test:{}".format(str(i + 1), train_total_loss, test_total_loss))
        np.savetxt('train_result.txt', train_result)
        np.savetxt('test_result.txt', test_result)

    np.savetxt("train_loss.txt", train_losses)
    np.savetxt("test_loss.txt", test_losses)

if __name__=='__main__':
    main()







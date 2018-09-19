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
from tensorflow.contrib.rnn import BasicLSTMCell,LSTMCell,GridLSTMCell
from tensorflow.contrib.grid_rnn import Grid2LSTMCell,Grid2BasicLSTMCell
from time import time

# os.environ['CUDA_VISIBLE_DEVICES']='0'  #当没有相应的GPU设备时，会使用CPU来运行。


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
    # normalize features
    # scaler = MinMaxScaler(feature_range=(0, 1))
    scaled=max_Min(values)
    # scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)
    # drop columns we don't want to predict
    reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
    # print(reframed)
    return reframed
####


def split_dataset_multiFeature(data):
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

def create_dataset_single_feature(data,timestep=1):
    """
    主要是分割单个特征的数据集
    :return:
    """
    data=np.loadtxt(data)
    # scaler=MinMaxScaler(feature_range=(0,1))
    # data=scaler.fit_transform(data)
    train_size=int(len(data)*0.7)
    test_size=len(data)-train_size
    data_X,data_Y=[],[]
    for i in range(len(data)-timestep-1):
        a=data[i:i+timestep]
        data_X.append(a)
        data_Y.append(data[i+timestep])
    # 将测试集和训练集存起来
    data_X=np.array(data_X)
    data_Y=np.array(data_Y)
    data_X=np.reshape(data_X,newshape=[-1,timestep,data_X.shape[1]])
    return data_X,data_Y

def split_train_test_dataset_single(dataset_X,dataset_Y,ratio=0.7,Is=True):
    """
    :param dataset:
    :return:
    """
    train_size=int(len(dataset_X)*ratio)
    train_X=dataset_X[0:train_size,:,:]

    train_dataset_X=dataset_X[0:train_size]
    train_dataset_Y=dataset_Y[0:train_size]

    test_dataset_X=dataset_X[train_size:]
    test_dataset_Y=dataset_Y[train_size:]

    test_X=dataset_X[train_size:,:,:]
    train_Y=dataset_Y[0:train_size]
    test_Y=dataset_Y[train_size:]
    if Is==True:
        np.savetxt("train_dataset_X.txt",train_dataset_X)
        np.savetxt("train_dataset_Y.txt",train_dataset_Y)

        np.savetxt("test_dataset_X.txt",test_dataset_X)
        np.savetxt("test_dataset_Y.txt",test_dataset_Y)

    return train_X,train_Y,test_X,test_Y

def add_label_to_CSV(test_dataset,ratio=0.05):
    """
    给测试集添加标签，对于不同的异常值情况进行分类,0-网络丢包异常，1-硬件异常，2-攻击异常，3-表示正常的值
    ,最后将其保存为csv文件格式的文件,这个数据集作为SVM模型的数据集，以及用来测试GridLSTM模型的
    :param test_dataset:测试集
    :param ratio:异常值占总数据集的比例
    :return:
    """
    test_dataset=np.loadtxt(test_dataset)
    size=int(len(test_dataset)*ratio)
    np.random.seed(0)   #设置随机种子，复现结果
    rand_index=list(np.random.choice(a=len(test_dataset),size=size,replace=False))
    columns=["value"]
    #添加一列，作为对应的类别
    data=pd.DataFrame(data=test_dataset,columns=columns)
    data["class"]=3   #表示正常的值
    print("未分配前所有的异常索引个数:",len(rand_index))
    #网络丢包异常点的分配
    for i,index in zip(range(round(size/3)),rand_index):
        data.iloc[index,0]=0      #网络丢包值为0，此时出现网络丢包异常
        data.iloc[index,1]=0
        rand_index.remove(index)
    print("网络丢包异常:",len(rand_index))
    #网络攻击异常点的分配
    for i,index in zip(range(round(size/3)),rand_index):
        data.iloc[index,0]=np.random.uniform(23,34,1)  #就是某一时间段我们将值设在23-34之间
        data.iloc[index,1]=1
        rand_index.remove(index)
    print("网络攻击异常点:",len(rand_index))
    del rand_index
    #硬件异常点的分配
    # 数据在某段时间一直趋于某一个值，突然又趋于另外一个值，此时出现了硬件异常，硬件异常最不好识别，容易把正常的值识别为异常值
    ran_index=np.arange(432,450,1)
    for index in ran_index:
        data.iloc[index,0]=np.random.uniform(37,38,1)
        data.iloc[index,1]=2
        # rand_index.remove(index)
    del ran_index
    ran_index=np.arange(720,740,1)
    for index in ran_index:
        data.iloc[index,0]=np.random.uniform(37,38,1)
        data.iloc[index,1]=2
    del ran_index
    ran_index=np.arange(3200,3220,1)
    for index in ran_index:
        data.iloc[index,0]=np.random.uniform(37,38,1)
        data.iloc[index,1]=2
    del ran_index
    print("硬件异常点:",58)
    data.to_csv("test_dataset.csv",sep=" ",index=None)   #将数据保存为csv格式的文件

def test_annoly_dataset(test_dataset,ratio=0.05):
    """
    返回带有异常值的测试集，这里将的异常值占总的数据集的1/10
    对于不同的异常值进行分类，主要是三类：0-丢包异常、1-硬件异常、2-攻击异常
    0：丢包异常，是传输值为0
    1：硬件异常，突然出现异常
    :param test_dataset:
    :return:
    """
    test_dataset=pd.read_csv(test_dataset)
    print(test_dataset)
    size=int(len(test_dataset)*ratio)
    print(size)
    rand_index=np.random.choice(a=len(test_dataset),size=size,replace=False)
    print(test_dataset)
    for i,index  in zip(range(0,int(len(rand_index)/3),1),rand_index):
        test_dataset.iloc[index,'value']=0
        test_dataset.iloc[index,'class']=0
    #对于异常值，我们将其标为0，1，2三类标签
        #产生硬件异常
        #产生攻击异常
    print(test_dataset)

class Config():
    """
    配置类
    """
    def __init__(self,train_x,train_y):
        self.timesteps=1
        self.features=1
        self.outputdims=1
        self.batch_size=50
        self.learning_rate=0.0001
        self.training_epoch=10
        self.keep_prob=0.9
        self.hidden_nums=9
        self.hidden_two=54
        self.hidden_three=27
        self.keep_prob=0.8

        self.W = {
            'hidden': tf.Variable(tf.random_normal([self.features, self.hidden_nums])),
            'output': tf.Variable(tf.random_normal([self.hidden_two, 1]))
        }
        self.biases = {
            'hidden': tf.Variable(tf.random_normal([self.hidden_nums])),
            'output': tf.Variable(tf.random_normal([1]))
        }

def GridLSTM_Model(input_data,config):
    """
    LSTM模型
    :param input_data:输入数据
    :param config:配置参数类对象
    :return:
    """
    input_data=tf.transpose(input_data,[1,0,2])
    input_data=tf.reshape(input_data,[-1,config.features])
    input_data=tf.nn.relu(tf.matmul(input_data,config.W['hidden'])+config.biases['hidden'])
    input_data=tf.split(input_data,config.timesteps,0)
    lstm_cell1=Grid2LSTMCell(num_units=config.hidden_two,tied=True,state_is_tuple=True)  #返回的是两个方向的值，怎么做？？？？？？？
    stack_lstm=tf.nn.rnn_cell.MultiRNNCell(cells=[lstm_cell1],state_is_tuple=True)   #返回的是元组形式，用的是一层GridLSTMCell
    outputs,_=tf.nn.static_rnn(cell=stack_lstm,inputs=input_data,dtype=tf.float32)
    outputs=tf.squeeze(outputs,0)  #进行压缩
    output=tf.nn.xw_plus_b(outputs[-1],config.W['output'],config.biases['output'])
    output=tf.matmul(outputs[-1],config.W['output'])+config.biases['output']
    return output,_

def main():
    # add_label_to_CSV("test_dataset_X.txt")
    # test_annoly_dataset("test_dataset.csv")
    data_X,data_Y=create_dataset_single_feature("data_light_new.txt")
    train_x,train_y,test_x,test_y=split_train_test_dataset_single(data_X,data_Y,Is=False)

    # train_x, train_y, test_x, test_y = split_dataset('Beijing_LSTM/pollution.csv')
    train_y = train_y.reshape([-1, 1])
    test_y = test_y.reshape([-1, 1])
    # print(train_y.shape)
    #
    # # 定义一些占位符与变量
    config = Config(train_x, train_y)
    X = tf.placeholder(dtype=tf.float32, shape=[None, config.timesteps, config.features])
    Y = tf.placeholder(dtype=tf.float32, shape=[None, config.outputdims])
    epoch = config.training_epoch
    lr = config.learning_rate
    batch_size = config.batch_size

    prediction_Y,_=GridLSTM_Model(X, config)
    tf.summary.histogram('predicton',prediction_Y)   #创建直方图的日志
    #利用MSE来做度量标准
    cost=tf.reduce_mean(tf.square(tf.subtract(prediction_Y,Y)))
    tf.summary.scalar('cost',tensor=cost)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
    # 生成saver
    saver=tf.train.Saver()
    init = tf.global_variables_initializer()
    sess = tf.Session()
    summary_writer = tf.summary.FileWriter("/log", sess.graph)
    sess.run(init)
    test_losses = []
    train_losses = []
    train_result_total=[]

    start=time()
    for i in range(epoch):
        train_total_loss = 0.0
        test_total_loss = 0.0
        # print("============epoch:",str(i+1),"======================")
        for start, end in zip(range(0, len(train_x), batch_size), range(batch_size, len(train_x) + 1, batch_size)):
            sess.run(optimizer, feed_dict={X: train_x[start:end], Y: train_y[start:end]})
            # saver.save(sess,save_path=saver_dir+"lstm_model.cpkt")
            loss_train, train_result = sess.run([cost, prediction_Y], feed_dict={X: train_x, Y: train_y})
            loss_test, test_result = sess.run([cost, prediction_Y], feed_dict={X: test_x, Y: test_y})
            test_total_loss += loss_test
            train_total_loss += loss_train
            train_result_total.append(train_result)
            np.savetxt('train_result_5.txt', train_result)
            np.savetxt('test_result_5.txt', test_result)
        test_losses.append(test_total_loss)
        train_losses.append(train_total_loss)
        print("epoch:{}======loss_train:{}===============loss_test:{}".format(str(i + 1), train_total_loss,test_total_loss))
        # summary_str=sess.run(merge_summmary_op,feed_dict={X:train_x,Y:train_y})
        # summary_writer.add_summary(summary_str,epoch)
    np.savetxt("train_loss.txt",train_losses)
    np.savetxt("test_loss.txt",test_losses)
    end=time()

    print("traing has been finished,it cost {}".format(end-start))

if __name__=='__main__':
    main()







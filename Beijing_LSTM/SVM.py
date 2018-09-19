"""-*- coding: utf-8 -*-
 DateTime   : 2018/9/10 21:27
 Author  : Peter_Bonnie
 FileName    : SVM.py
 Software: PyCharm
"""
import  numpy as np
import pandas as pd
from sklearn.svm import libsvm
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from math import fabs
from sklearn.metrics import accuracy_score,auc,recall_score,f1_score,precision_score,roc_curve,roc_auc_score
from sklearn import  metrics
import tensorflow as tf
import  matplotlib.pyplot as plt

"""
主要是使用的SVM实现多分类，主要有两种方法：1.一对一方法,当K类时，需要创建k(k-1)/2个分类器 
                                    2.还有一种一对多方法，为每个类别创建一个高斯核函数分类器，最后的预测类别是具有最大SVM间隔的类别
"""

#对数据集进行打标签
def create_error_dataset(true_dataset,predict_dataset,Is=True):
    """
    创建误差数据集
    :param true_dataset:
    :param predict_dataset:
    :return:
    """
    if Is==True:
        error_dataset=[]
        true_dataset=np.loadtxt(true_dataset).astype(dtype=np.float32)
        predict_dataset=np.loadtxt(predict_dataset).astype(dtype=np.float32)
        for i in range(len(predict_dataset)):
            error_dataset.append(fabs(true_dataset[i]-predict_dataset[i]))
        np.savetxt("error_dataset.txt",error_dataset)
    return error_dataset

#加一列表示对应的标签，然后根据这个标签来计算相应分类的准确率，精确率....
#这个仅针对测试训练集的，我们仅仅需要将测试集上
#对于数据集上的不同
#0,1,2,3
#现在做的都是时间步长为1的实验

"""
1.误差数据集主要的构建方法以及如何使用SVM模型怎么运用训练集进行评估
2.先是对原始测试数据集进行标注，哪些是异常值，哪些是正常值。
"""
#运用SVM模型来做4分类的预测
def SVM_Model(dataset_X,dataset_Y):
    """
    返回预测的效果
    :param dataset_X:
    :param dataset_Y:
    :return:
    """
    # train_X,train_Y,test_X,test_Y=split_train_test_dataset(dataset_X=dataset_X,dataset_Y=dataset_Y)
def crate_data_to_target(train_X,train_Y,test_X,test_Y):
    """
    主要是计算各个类别的情况，创造相应的数据集
    :param train_X:
    :param train_Y:
    :param test_X:
    :param test_Y:
    :return:
    """
    #训练集
    train_data_X=np.array([x for x in train_X])
    train_data_Y1=np.array([1 if y==0 else -1 for y in train_Y])
    train_data_Y2=np.array([1 if y==1 else -1 for y in train_Y])
    train_data_Y3=np.array([1 if y==2 else -1 for y in train_Y])
    train_data_Y4=np.array([1 if y==3 else -1 for y in train_Y])
    train_data_Y=np.array([train_data_Y1,train_data_Y2,train_data_Y3,train_data_Y4])

    #测试集
    test_data_X=np.array([x for x in test_X])
    test_data_Y1=np.array([1 if y==0 else -1 for y in test_Y])
    test_data_Y2=np.array([1 if y==1 else -1 for y in test_Y])
    test_data_Y3=np.array([1 if y==2 else -1 for y in test_Y])
    test_data_Y4=np.array([1 if y==3 else -1 for y in test_Y])
    test_data_Y=np.array([test_data_Y1,test_data_Y2,test_data_Y3,test_data_Y4])

    print(train_data_Y)

if __name__=="__main__":
    data_X=pd.read_csv("test_value.csv")
    data_Y=pd.read_csv("test_label.csv")
    train_X,test_X,train_Y,test_Y=train_test_split(data_X,data_Y,test_size=0.7,random_state=20)
    crate_data_to_target(train_X,train_Y,test_X,test_Y)

    # add_label("error_dataset.txt")






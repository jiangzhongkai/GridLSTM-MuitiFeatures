"""-*- coding: utf-8 -*-
 DateTime   : 2018/9/10 21:27
 Author  : Peter_Bonnie
 FileName    : SVM.py
 Software: PyCharm
"""
import  numpy as np
import pandas as pd
from sklearn.metrics.scorer import f1_score,recall_score,precision_score,accuracy_score
from sklearn.svm import libsvm
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from math import fabs

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

"""
1.误差数据集主要的构建方法以及如何使用SVM模型怎么运用训练集进行评估
2.先是对原始测试数据集进行标注，哪些是异常值，哪些是正常值。
"""
def add_label(error_test_dataset):
    """
    添加一列作为标签
    :param dataset:
    :return:
    """
    error_dataset=np.loadtxt(error_test_dataset).astype(np.float32)
    print(error_dataset)

    for  i in range(len(error_dataset)):




def Min_Max_Scaler():
    """
    进行数据归一化
    :return:
    """
    pass


def SVM_Model():
    """
    SVM模型
    :return:
    """

if __name__=="__main__":
    # error_dataset=create_error_dataset("data_light.txt","train_result.txt",Is=False)
    add_label("error_dataset.txt")






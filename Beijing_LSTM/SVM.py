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
import csv

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
    # class_0_x=[x for i,x in enumerate(train_data_X) if train_Y[i]==0]
    # class_1_x=[x for i,x in enumerate(train_data_X) if train_Y[i]==1]
    # class_2_x=[x for i,x in enumerate(train_data_X) if train_Y[i]==2]
    # class_3_x=[x for i,x in enumerate(train_data_X) if train_Y[i]==3]
    return train_data_X,train_data_Y,test_data_X,test_data_Y

def open_csv(test_value,test_label):
    """
    :param test_value:
    :param test_label:
    :return:
    """
    with open(test_value,"r") as f:
        reader=csv.reader(f)
        data_X=[float(row[0]) for row in reader]
    with open(test_label,"r") as f:
        reader=csv.reader(f)
        data_Y=[int(row[0]) for row in reader]
    return data_X,data_Y

def reshape_mat(mat,batch_size):
    v1=tf.expand_dims(mat,1)
    v2=tf.reshape(v1,[4,batch_size,1])
    return (tf.matmul(v2,v1))
if __name__=="__main__":
    data_X,data_Y=open_csv("test_value.csv","test_label.csv")
    train_X,test_X,train_Y,test_Y=train_test_split(data_X,data_Y,test_size=0.7,random_state=20)
    train_data_X,train_data_Y,test_data_X,test_data_Y=crate_data_to_target(train_X,train_Y,test_X,test_Y)
    #定义一些常量和占位符
    batch_size = 20
    X_data=tf.placeholder(shape=[None,1],dtype=tf.float32)
    Y_target=tf.placeholder(shape=[4,None],dtype=tf.float32)
    prediction_grid=tf.placeholder(shape=[None,1],dtype=tf.float32)
    b=tf.Variable(tf.random_normal(shape=[4,batch_size],dtype=tf.float32))

    #计算高斯核函数
    gamma=tf.constant(-10.0)
    dist=tf.reduce_sum(tf.square(X_data),1)
    dist=tf.reshape(dist,[-1,1])
    sq_dists=tf.add(tf.subtract(dist,tf.multiply(2.,tf.matmul(X_data,tf.transpose(X_data)))),tf.transpose(dist))
    my_kernel=tf.exp(tf.multiply(gamma,tf.abs(sq_dists)))

    model_output=tf.matmul(b,my_kernel)
    first_term=tf.reduce_sum(b)
    b_vec_cross=tf.matmul(tf.transpose(b),b)
    Y_target_cross=reshape_mat(Y_target,batch_size)

    second_term=tf.reduce_sum(tf.multiply(my_kernel,tf.multiply(b_vec_cross,Y_target_cross)),[1,2])
    loss=tf.reduce_sum(tf.negative(tf.subtract(first_term,second_term)))

    #创建预测的核函数
    rA=tf.reshape(tf.reduce_sum(tf.square(X_data),1),[-1,1])
    rB=tf.reshape(tf.reduce_sum(tf.square(prediction_grid),1),[-1,1])
    pred_sq_dist=tf.add(tf.subtract(rA,tf.multiply(2.,tf.matmul(X_data,tf.transpose(prediction_grid)))),tf.transpose(rB))
    pred_kernel=tf.exp(tf.multiply(gamma,tf.abs(pred_sq_dist)))

    #创建预测函数
    prediction_output=tf.matmul(tf.multiply(Y_target,b),pred_kernel)
    prediction=tf.arg_max(prediction_output-tf.expand_dims(tf.reduce_mean(prediction_output,1),1),0)
    accuracy=tf.reduce_mean(tf.cast(tf.equal(prediction,tf.argmax(Y_target,0)),tf.float32))

    #初始化相关变量
    opt=tf.train.GradientDescentOptimizer(0.01)
    train_step=opt.minimize(loss)

    init=tf.global_variables_initializer()
    sess=tf.Session()
    sess.run(init)
    loss_vec=[]
    batch_accuracy=[]
    for i in range(1000):
        rand_index=np.random.choice(len(train_data_X),size=batch_size)
        rand_x=train_data_X[rand_index].reshape([batch_size,-1])
        rand_y=train_data_Y[:,rand_index]
        sess.run(train_step,feed_dict={Y_target:rand_y,X_data:rand_x})
        temp_loss,temp_accuracy,pred_output=sess.run([loss,accuracy,prediction],feed_dict={X_data:rand_x,Y_target:rand_y,prediction_grid:rand_x})
        loss_vec.append(temp_loss)
        batch_accuracy.append(temp_accuracy)
        if (i+1)%2==0:
            # print(rand_y.shape)
            true_label=[]
            for p in range(rand_y.shape[1]):
                for j in range(rand_y.shape[0]):
                    if rand_y[j][p]==1:
                        true_label.append(j)
                        continue
            print("true_value:",true_label)
            print("pred_value:",pred_output)
        print("epoch:{}=======loss:{:.9f}========accuracy:{:.9f}".format(str(i+1),temp_loss,temp_accuracy))

    #metrics
    test_data_X=np.reshape(test_data_X,[-1,1])
    print(test_data_X.shape)
    print(test_data_Y.shape)  #这里报错的原因是形状不匹配，所以我需要重新定义形状
    val_accuracy,y_pred=sess.run([accuracy,prediction],feed_dict={X_data:test_data_X,Y_target:test_data_Y,prediction_grid:test_data_X})
    print("validation accuracy:",val_accuracy)
    y_true=np.argmax(test_Y,1)
    print("precision:",precision_score(y_true,y_pred))

    plt.figure()
    plt.plot(batch_accuracy, 'r--', label='Accuracy',lw=3)
    plt.title('Batch Accuracy')
    plt.xlabel('Generation')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()






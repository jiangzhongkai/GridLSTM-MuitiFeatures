"""-*- coding: utf-8 -*-
 DateTime   : 2018/10/14 19:03
 Author  : Peter_Bonnie
 FileName    : test_SVM.py
 Software: PyCharm
"""
import numpy as np
import tensorflow as tf
from SVM import *
from sklearn.metrics import accuracy_score,auc,recall_score,f1_score,precision_score,roc_curve,roc_auc_score,confusion_matrix

test_data_X=np.loadtxt("svm_test_x.txt",dtype=np.float64)
test_data_Y=np.loadtxt("svm_test_y.txt",dtype=np.float64)


#定义一些常量和占位符
batch_size = 100
X_data=tf.placeholder(shape=[None,1],dtype=tf.float32)
Y_target=tf.placeholder(shape=[4,None],dtype=tf.float32)
prediction_grid=tf.placeholder(shape=[None,1],dtype=tf.float32)
b=tf.Variable(tf.random_normal(shape=[4,batch_size],dtype=tf.float32))

# 计算高斯核函数
gamma = tf.constant(-10.0)
dist = tf.reduce_sum(tf.square(X_data), 1)
dist = tf.reshape(dist, [-1, 1])
sq_dists = tf.add(tf.subtract(dist, tf.multiply(2., tf.matmul(X_data, tf.transpose(X_data)))), tf.transpose(dist))
my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))

model_output = tf.matmul(b, my_kernel)
first_term = tf.reduce_sum(b)
b_vec_cross = tf.matmul(tf.transpose(b), b)
Y_target_cross = reshape_mat(Y_target, batch_size)

second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, Y_target_cross)), [1, 2])
loss = tf.reduce_sum(tf.negative(tf.subtract(first_term, second_term)))

# 创建预测的核函数
rA = tf.reshape(tf.reduce_sum(tf.square(X_data), 1), [-1, 1])
rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1), [-1, 1])
pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(X_data, tf.transpose(prediction_grid)))),
                      tf.transpose(rB))
pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

# 创建预测函数
prediction_output = tf.matmul(tf.multiply(Y_target, b), pred_kernel)
prediction = tf.arg_max(prediction_output - tf.expand_dims(tf.reduce_mean(prediction_output, 1), 1), 0)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(Y_target, 0)), tf.float32))

# 初始化相关变量
opt = tf.train.GradientDescentOptimizer(0.001)
train_step = opt.minimize(loss)


#加载模型
sess=tf.Session()
saver=tf.train.Saver()
saver.restore(sess,"./SVM_Model/SVM.ckpt")

test_data_X=np.reshape(test_data_X,[-1,1])
max_score={
    'f1_score': 0.0,
    'accuracy': 0.0,
    'precision': 0.0,
    'Recall': 0.0,
    'index': 0
}
#利用测试集数据来评估模型分类的好坏
for i in range(100):
    print("================epoch:{}==========================".format(str(i + 1)))
    rand_test_index = np.random.choice(len(test_data_X), size=batch_size, replace=False)
    rand_test_x = test_data_X[rand_test_index]
    rand_test_y = test_data_Y[:,rand_test_index]
    val_accuracy, y_pred = sess.run([accuracy, prediction], feed_dict={X_data: rand_test_x, Y_target: rand_test_y,
                                                                       prediction_grid: rand_test_x})
    print("validation accuracy:", val_accuracy)
    y_true = []
    for p in range(rand_test_y.shape[1]):
        for j in range(rand_test_y.shape[0]):
            if rand_test_y[j][p] == 1:
                y_true.append(j)
                continue
    y_true = np.array(list(y_true))
    y_pred = np.array(list(y_pred))

    print("y_ture=====", y_true)
    print("y_pred====", y_pred)
    print("accuracy:", accuracy_score(y_true, y_pred))
    print("precision:", precision_score(y_true, y_pred, average='macro'))
    print("Recall:", recall_score(y_true, y_pred, average='macro'))
    print("f1_score:", f1_score(y_true, y_pred, average='macro'))
    # print("confusion matrix:",confusion_matrix(y_true,y_pred))
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    print("auc:", auc(fpr, tpr))
    if max_score['f1_score']<f1_score(y_true, y_pred, average='macro'):
        max_score['f1_score']=f1_score(y_true, y_pred, average='macro')
        max_score['accuracy']=accuracy_score(y_true, y_pred)
        max_score['precision']=precision_score(y_true, y_pred, average='macro')
        max_score['Recall']=recall_score(y_true, y_pred, average='macro')
        max_score['index']=i-1
    # print("tpr:{},fpr:{},thresholds:{}".format(fpr,tpr,thresholds))
print(max_score)


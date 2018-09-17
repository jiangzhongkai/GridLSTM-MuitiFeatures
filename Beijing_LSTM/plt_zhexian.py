"""-*- coding: utf-8 -*-
 DateTime   : 2018/9/3 13:37
 Author  : Peter_Bonnie
 FileName    : plt_zhexian.py
 Software: PyCharm
"""
import numpy as np
import matplotlib.pyplot as plt
import math

true_dataset=np.loadtxt("data_light_new.txt")[100:400]
predict_dataset=np.loadtxt("train_result_2.txt")[100:400]


true_dataset_test=np.loadtxt("data_light_new.txt")[-300:]
predict_dataset_test=np.loadtxt("test_result_2.txt")[-300:]

error_dataset=[]
error_dataset_test=[]
for  i in range(len(predict_dataset)):
    error_dataset.append(math.fabs(predict_dataset[i]-true_dataset[i]))

for i in range(len(predict_dataset_test)):
    error_dataset_test.append(math.fabs(predict_dataset_test[i]-true_dataset_test[i]))

plt.figure(figsize=(12,10))
plt.subplot(221)
plt.title("train_true")
plt.plot(true_dataset,'r-',lw=2,label='true_value')
plt.plot(predict_dataset,'g--',lw=2,label='predict_value')
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.legend(loc="best")

plt.subplot(222)
plt.title("test_true")
plt.plot(true_dataset_test,'r-',lw=2,label='true_value')
plt.plot(predict_dataset_test,'g--',lw=2,label='predict_value')
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.legend(loc="best")

plt.subplot(223)
plt.title("train_error")
plt.xlabel("Epochs")
plt.plot(error_dataset,'b--',lw=2,label='test_error')
plt.legend(loc="best")

plt.subplot(224)
plt.title("test_error")
plt.xlabel("Epochs")
plt.plot(error_dataset_test,'b--',lw=2,label="test_error")
plt.legend(loc="best")

plt.savefig("true_predict_value.jpg")
plt.show()
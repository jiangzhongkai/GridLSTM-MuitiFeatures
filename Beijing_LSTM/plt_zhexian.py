"""-*- coding: utf-8 -*-
 DateTime   : 2018/9/3 13:37
 Author  : Peter_Bonnie
 FileName    : plt_zhexian.py
 Software: PyCharm
"""
import numpy as np
import matplotlib.pyplot as plt

train=np.loadtxt('train_loss.txt')
test=np.loadtxt('test_loss.txt')

plt.figure()
plt.plot(train,'r-',label='train_loss',lw=2)
plt.plot(test,'b--',label='test_loss',lw=2)
plt.xlabel('Generations')
plt.ylabel('Losses')
plt.grid(ls=':')
plt.legend(loc='best')
plt.show()
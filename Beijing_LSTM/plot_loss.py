"""-*- coding: utf-8 -*-
 DateTime   : 2018/9/19 10:39
 Author  : Peter_Bonnie
 FileName    : plot_loss.py
 Software: PyCharm
"""
import matplotlib.pyplot as plt
import numpy as np

LSTM_1=np.loadtxt("1_LSTM_loss.txt")
LSTM_2=np.loadtxt("2_LSTM_loss.txt")
Grid_1=np.loadtxt("1_GLSTM_loss.txt")
Grid_2=np.loadtxt("2_GLSTM_loss.txt")

plt.figure(figsize=(8,6))
plt.plot(LSTM_1,'r-',label='LSTM_Train_Loss',lw=3)
plt.plot(LSTM_2,'g--',label='LSTM_Test_Loss',lw=3)
plt.plot(Grid_1,'b-',label='GLSTM_Train_Loss',lw=3)
plt.plot(Grid_2,'r--',label='GLSTM_Test_Loss',lw=3)
plt.legend(loc="best")
plt.xlabel('Epochs')
plt.ylabel("Loss")
plt.grid(ls=':')
plt.savefig('loss_compare.jpg')
plt.show()
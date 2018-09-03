"""-*- coding: utf-8 -*-
 DateTime   : 2018/9/2 16:23
 Author  : Peter_Bonnie
 FileName    : plt_fig
 Software: PyCharm
"""
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('pollution.csv',header=0,index_col=0)
values=dataset.values
groups=[0,1,2,3,4,5,6,7]

i=1
plt.figure()
for group in groups:
    plt.subplot(len(groups),1,i)
    plt.plot(values[:,group])
    plt.title(dataset.columns[group],y=0.5,loc='right')
    i+=1
plt.savefig('fig.pdf')
plt.show()




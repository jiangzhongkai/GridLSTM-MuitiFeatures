"""
利用谷歌深度学习框架Tensorflow和Stack LSTM进行多特征数据的预测
"""
Tensorflow 1.8.0
Python 3.5

需要的第三方库：
numpy
matplotlib
pandas
sklearn
...

主要实现流程：
step_1:对数据集进行处理，并处理成LSTM的输入格式[samples,timesteps,features]
step_2: 1)tf.nn.rnn_cell.BasicLSTMCell()创建基本的LSTM单元
        2)tf.nn.rnn_cell.MultiRNNCell() 创建堆栈式的LSTM结构
        3)tf.nn.rnn_cell.MultiRNNCell().zero_state()   #主要是用于初始化状态
        4)tf.nn.static_rnn()   #获取最后一个隐藏层单元的输出
step_3:利用已构建的模型进行训练
step_4:利用测试集来对模型进行评估，这里我选用的是RMSE作为度量标准
step_5:进行实验分析与比较


#这里主要是将时间步设为1的结果
#可以将时间步设为其他情况来做分析
       

"""
利用谷歌深度学习框架Tensorflow和Stack LSTM进行多特征数据的预测
"""
step_1:做数据集的处理
step_2:LSTM的建模：1.tf.nn.rnn_cell.BasicLSTMCell()
                  2.如果需要多个LSTM进行堆栈,则还需要tf.nn.rnn_cell.MultiRNNCell()
                  3.tf.zero_state()
                  4.tf.nn.rnn.static_rnn()   或者   tf.nn.rnn.dynamic_rnn()
                  5.最后获得最后一个单元的输出
step_3:利用建模的模型进行预测并利用评估方法进行模型的评估
step_4:实验结果的分析
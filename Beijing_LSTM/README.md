
##### 利用谷歌深度学习框架Tensorflow和Grid LSTM进行多特征数据的预测
|需要的第三方库|
|----|
|Tensorflow 1.8.0|
|Python 3.5|
|numpy|
|matplotlib|
|pandas|
|sklearn|


##### 主要实现流程：
- step_1:对数据集进行处理，并处理成LSTM的输入格式[samples,timesteps,features]
- step_2: 
        - 1)tf.nn.rnn_cell.BasicLSTMCell()创建基本的LSTM单元
        - 2)tf.nn.rnn_cell.MultiRNNCell() 创建堆栈式的LSTM结构
        - 3)tf.nn.rnn_cell.MultiRNNCell().zero_state()   #主要是用于初始化状态
        - 4)tf.nn.static_rnn()   #获取最后一个隐藏层单元的输出
- step_3:利用已构建的模型进行训练
- step_4:利用测试集来对模型进行评估，这里我选用的是RMSE作为度量标准
- step_5:进行实验分析与比较

##### 评估指标：
- 在分类模型上，我们主要采用的是
```math
f_{1}score,accuracy,precision,recall,roc,auc
```
- 在回归模型上主要采用的是
```math  
MAPE,R^{2}
```


##### 数据集：
   - 训练集：没有异常值
   - 验证集：没有异常值
   - 测试集：异常值，通过异常值来构造相应的SVM模型，并对模型进行评估






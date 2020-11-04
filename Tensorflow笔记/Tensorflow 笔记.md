# Tensorflow 笔记
Author: qianS

Email: qiangezaici@outlook.com

date: 2020 - 10 - 23

---
### 创建张量

- 直接构造Tensor数据类型

  > tf.constant(张量内容, dtype=数据类型(可选))

  ```python
  import tensorflow as tf
  
  a = tf.constant([1,5], dtype = tf.int64)
  print(a)
  print(a.dtype)
  print(a.shape)
  #运行结果
  #tf.Tensor([1 5], shape(2,), dtype=int64)
  #<dtype: 'int64'>
  #(2,)
  ```

- 由numpy数据类型转换为Tensor数据类型

  > tf.convert_to_tensor(数据名, dtype=数据类型(可选))

```python
import tensorflow as tf
import numpy as np

a = np.arange(0, 5)
b = tf.convert_to_tensor(a, dtype = tf.int64)
print(a)
print(b)
#运行结果
#[0 1 2 3 4]
#tf.Tensor([0 1 2 3 4], shape(5,), dtype=int64)
```

- 创建全为0的张量

  > tf.zeros(维度)

- 创建全为1的张量

  > tf.ones(维度)

- 创建全为指定值的张量

  > tf.fill(维度, 指定值)

```python
import tensorflow as tf

a = tf.zeros([2, 3])
b = tf.ones(4)
c = tf.fill([2, 2], 9)
print(a)
print(b)
print(c)
#运行结果
#tf.Tensor([[0. 0. 0.] [0. 0. 0.]], shape=(2, 3), dtype=float32)
#tf.Tensor([1. 1. 1. 1.], shape=(4,), dtype=float32)
#tf.Tensor([[9 9] [9 9]], shape=(2, 2), dtype=int32)
```

- 生成正态分布的随机数, 默认均值为0, 标准差为1

  > tf.random.normal(维度, mean = 均值, stddev = 标准差)

- 生成截断式正态分布的随机数

  > tf.random.truncated_normal(维度, mean = 均值, stddev = 标准差)

- 生成均值分布随机数 [minval, maxval) 左闭右开区间

  > tf.random.uniform(维度, minval = 最小值, maxval = 最大值)

```python
import tensorflow as tf

d = tf.random.normal([2, 2], mean = 0.5, stddev = 1)
print(d)

e = tf.random.truncated_normal([2, 2], mean = 0.5, stddev = 1)
print(e)

f = tf.random.uniform([2, 2], minval = 0, maxval = 1)
print(f)
#运行结果
#tf.Tensor(
#[[-0.7145612   0.30294186]
# [ 1.4073212   0.4383824 ]], shape=(2, 2), dtype=float32)
#
#tf.Tensor(
#[[ 0.10782576 -0.7288412 ]
# [-0.37378424  0.29531428]], shape=(2, 2), dtype=float32)
#
#tf.Tensor(
#[[0.23910916 0.09170806]
# [0.46913362 0.03215027]], shape=(2, 2), dtype=float32)
```

- 强制tensor转换为该数据类型

  > tf.cast(张量名, dtype = 数据类型)

- 计算张量维度上元素的最小值

  > tf.reduce_min(张量名)

- 计算张量维度上元素的最大值

  > tf.reduce_max(张量名)

```python
import tensorflow as tf

x1 = tf.constant([1., 2., 3.],
                dtype = tf.float64)
print(x1)

x2 = tf.cast(x1, tf.int32)
print(x2)

print(tf.reduce_min(x2),
     tf.reduce_max(x2))
#运行结果
#tf.Tensor([1. 2. 3.], shape=(3,), dtype=float64)
#tf.Tensor([1 2 3], shape=(3,), dtype=int32)
#tf.Tensor(1, shape=(), dtype=int32) tf.Tensor(3, shape=(), dtype=int32)
```

- axis

  > axis = 0 表示跨行(维度, down), axis = 1 表示跨列(维度, across)
  >
  > 如果不指定axis, 则所有元素参与计算

- 计算张量沿着指定维度的平均值

  > tf.reduce_mean(张量名, axis = 操作轴)

- 计算张量沿着指定维度的和

  > tf.reduce_sum(张量名, axis = 操作轴)

```python
import tensorflow as tf

x = tf.constant([1, 2, 3], [2, 2, 3])
print(x)
print(tf.reduce_mean(x))
print(tf.reduce_sum(x, axis = 1))
#运行结果
#tf.Tensor([[1 2 3] [2 2 3]], shape=(2, 3), dtype=int32)
#tf.Tensor(2, shape=(), dtype=int32)
#tf.Tensor([6 7], shape=(2,), dtype=int32)
```

- 将变量标记为"可训练", 被标记的变量会在反向传播中记录梯度信息.神经网络训练中, 常用该函数标记带训练参数

  > tf.Variable(初始值)

```python
w = tf.Variable(tf.random.normal([2, 2], mean = 0, stddev = 1))
```

- 对应元素的四则运算(只有维度相同才能做四则运算)

  > tf.add(张量1, 张量2)           //加
  >
  > tf.subtract()(张量1, 张量2) //减
  >
  > tf.multiply(张量1, 张量2)   //乘
  >
  > tf.divide(张量1, 张量2)       //除

  ```python
  import tensorflow as tf
  
  a = tf.ones([1, 3])
  b = tf.fill([1, 3], 3.)
  
  print(a)
  print(b)
  
  print(tf.add(a, b))
  print(tf.subtract(a, b))
  print(tf.multiply(a, b))
  print(tf.divide(b, a))
  #运行结果
  #tf.Tensor([[1. 1. 1.]], shape=(1, 3), dtype=float32)
  #tf.Tensor([[3. 3. 3.]], shape=(1, 3), dtype=float32)
  #tf.Tensor([[4. 4. 4.]], shape=(1, 3), dtype=float32)
  #tf.Tensor([[-2. -2. -2.]], shape=(1, 3), dtype=float32)
  #tf.Tensor([[3. 3. 3.]], shape=(1, 3), dtype=float32)
  #tf.Tensor([[3. 3. 3.]], shape=(1, 3), dtype=float32)
  ```

  

- 平方, 次方与开方

  > tf.square(张量名)
  >
  > tf.pow(张量名, n次方数)
  >
  > tf.sqrt(张量名)

  ```python
  import tensorflow as tf
  
  a = tf.fill([1, 2], 3.)
  
  print(a)
  
  print(tf.pow(a, 3))
  print(tf.square(a))
  print(tf.sqrt(a))
  #运行结果
  #tf.Tensor([[3. 3.]], shape=(1, 2), dtype=float32)
  #tf.Tensor([[27. 27.]], shape=(1, 2), dtype=float32)
  #tf.Tensor([[9. 9.]], shape=(1, 2), dtype=float32)
  #tf.Tensor([[1.7320508 1.7320508]], shape=(1, 2), dtype=float32)
  ```

  

- 矩阵乘

  > tf.matmul(张量名)

  ```python
  import tensorflow as tf
  
  a = tf.ones([3, 2])
  b = tf.fill([2, 3], 3.)
  
  print(tf.matmul(a, b))
  #运行结果
  #tf.Tensor([[6. 6. 6.] [6. 6. 6.] [6. 6. 6.]], shape=(3, 3), dtype=float32)
  ```

- 切分传入张量的第一维度,生成输入特征/标签对,构建数据集

  > data = tf.data.Dataset.from_tensor_slices((输入特征, 标签))

  ```python
  import tensorflow as tf
  
  features = tf.constant([12, 23, 10, 17])
  labels = tf.constant([0, 1, 1, 0])
  dataset = tf.data.Dataset.from_tensor_slices((features, labels))
  
  print(dataset)
  
  for element in dataset:
      print(element)
  #运行结果
  #<TensorSliceDataset shapes: ((), ()), types: (tf.int32, tf.int32)>
  #
  #(<tf.Tensor: shape=(), dtype=int32, numpy=12>, <tf.Tensor: shape=(), dtype=int32, numpy=0>)
  #(<tf.Tensor: shape=(), dtype=int32, numpy=23>, <tf.Tensor: shape=(), dtype=int32, numpy=1>)
  #(<tf.Tensor: shape=(), dtype=int32, numpy=10>, <tf.Tensor: shape=(), dtype=int32, numpy=1>)
  #(<tf.Tensor: shape=(), dtype=int32, numpy=17>, <tf.Tensor: shape=(), dtype=int32, numpy=0>)
  ```

- with结构记录计算过程, gradient求出张量梯度

  > with tf.GradientType() as tape:
  >
  > ​    若干个计算过程
  >
  > grad = tape.Gradient(函数, 对谁求导)

  ```python
  import tensorflow as tf
  
  with tf.GradientTape() as tape:
      w = tf.Variable(tf.constant(3.0))
      loss = tf.pow(w, 2)
  grad = tape.gradient(loss, w)
  print(grad)
  #运行结果
  #tf.Tensor(6.0, shape=(), dtype=float32)
  ```

- 独热编码, tf.one_hot

  > 在分类问题中, 常用独热码做标签, 标记类别: 1表示是, 0表示非.
  >
  > ​             (0狗尾草鸢尾    1杂色鸢尾     2弗吉尼亚鸢尾)
  >
  > 标签     :          1
  >
  > 独热码 : (         0.                    1.                        0.           )
  >
  > tf.one_hot(待转换数据, depth = 几分类)

  ```python
  import tensorflow as tf
  
  classes = 3
  labels = tf.constant([1, 0, 2]) # 输入的元素最小值为0, 最大为2
  output = tf.one_hot(labels, depth = classes)
  print(output)
  #运行结果
  #tf.Tensor([[0. 1. 0.] [1. 0. 0.] [0. 0. 1.]], shape=(3, 3), dtype=float32)
  ```

- tf.nn.softmax(x)使每个输出符合概率分布

  ![Softmax函数](.\Softmax函数.png)
  
- 自减操作

  > w.assign_sub(w要自减的内容)
  >
  > > 赋值操作, 更新参数的值并返回
  > >
  > > 调用assign_sub前, 先用tf.Variable定义变量w为可训练

  ```python
  import tensorflow as tf
  
  w = tf.Variable(4)
  w.assign_sub(1)
  print(w)
  #运行结果
  #<tf.Variable 'Variable:0' shape=() dtype=int32, numpy=3>
  ```

- 返回张量沿指定维度最大值的索引

  > tf.argmax(张量名, axis = 操作轴)

  ```python
  import tensorflow as tf
  import numpy as np
  
  test = np.array([[1, 2, 3], [2, 3, 4], [5, 4, 3], [8, 7, 2]])
  print(test)
  print(tf.argmax(test, axis = 0)) #返回每一列(经度)最大值的索引
  print(tf.argmax(test, axis = 1)) #返回每一行(纬度)最大值的索引
  #运行结果
  #[[1 2 3]
  # [2 3 4]
  # [5 4 3]
  # [8 7 2]]
  #
  #tf.Tensor([3 3 1], shape=(3,), dtype=int64)
  #tf.Tensor([2 2 0 0], shape=(4,), dtype=int64)
  ```

### 数据处理(鸢尾花)

- 数据读入

  > 通过sklearn包中的datasets直接下载数据集

```python
from sklearn import datasets
from pandas import DataFrame
import pandas as pd

x_data = datasets.load_iris().data   #返回iris数据集所有输入特征
y_data = datasets.load_iris().target #返回iris数据集所有标签
print("x_data from datasets: \n", x_data)
print("y_data from datasets: \n", y_data)

x_data = DataFrame(x_data, columns = ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']) #增加可读性, 使数据变为表格形式. 每一列增加中文标签
pd.set_option('display.unicode.east_asian_width', True) #设置列名对齐
print('x_data add a column: \n', x_data)

x_data['类别'] = y_data #新增一列, 列标签为'类别', 数据为y_data
print("x_data add a column: \n", x_data)
```

- 数据集乱序

  > 使用相同的seed, 使输入特征/标签一一对应

  ```python
  np.random.seed(116)
  np.random.shuffle(x_data)
  np.random.seed(116)
  np.random.shuffle(y_data)
  tf.random.set_seed(116)
  ```

- 拆分训练集和测试集

  > 本例使用后30个数据作为测试集

  ```python
  x_train = x_data[:-30]
  y_train = y_data[:-30]
  x_text = x_data[-30:]
  y_text = y_data[-30:]
  ```

- 配对[输入特征, 标签]对, 每次喂入一小撮(batch)

  ```python
  train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
  test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
  ```
- 定义网络中可训练参数, seed现实实际使用时不写

  ```python
  x1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev = 0.1, seed = 1)) #一层网络,四个输入特征, 3个输出特征(4行3列的张量)
  b1 = tf.Variable(tf.random.turncated_normal([3], stddev = 0.1, seed = 1))    #b1必须与w1维度一致
  ```

- 更新参数

  ```python
  for epoch in range(epoch):                                #数据集级别迭代
      for step, (x_train, y_train) in enumerate(train_db):  #batch级别迭代
          with tf.GradientTape() as tape:                   #记录梯度信息
              #前向传播过程计算y
              #计算总loss
          grads = tape.gradient(loss, [w1, b1])
          w1.assign_sub(lr * grads[0])                      #参数自更新
          b1.assign_sub(lr * grads[1])
      print("Epoch {}, loss: {}".format(epoch, loss_all/4)) #训练集尺寸120, batch尺寸32, 需循环4次
  ```

- 计算当前参数前向传播后的准确率, 显示当前acc

  ```python
  for x_test, y_test in test_db:
      y = tf.matmul(h, w) + b                    #y为预测结果
      y = tf.nn.softmax(y)                       #y符合概率分布
      pred = tf.argmax(y, axis = 1)              #返回y中最大值的索引, 即预测的分类
      pred = tf.cast(pred, dtype = y_test.dtype) #调整数据类型与标签一致
      correct = tf.cast(tf.equal(pred, y_test), dtype = tf.int32)
      correct = tf.reduce_sum(corrent)           #将每个batch的correct数加起来
      total_correct += int(corrent)              #将所有batch中的corrent数加起来
      total_number += x_test.shape[0]
  acc = total_correct / total_number
  print("test_acc: ", acc)
  ```

- acc / loss可视化

  ```python
  plt.title('Acc Curve')                   #图片标题
  plt.xlabel('Epoch')                      #x轴名称
  plt.ylabel('Acc')                        #y轴名称
  plt.plot(test_acc, label = "$Accuracy$") #逐点画出test_acc值并连线
  plt.legend()
  plt.show()
  ```

### 鸢尾花完整代码

```python
import tensorflow as tf
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np

x_data = datasets.load_iris().data  # 返回iris数据集所有输入特征
y_data = datasets.load_iris().target  # 返回iris数据集所有标签

np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

# 转换x的数据类型, 否则后面矩阵相乘时会因数据类型不一致报错
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

lr = 0.1  # 学习率
train_loss_results = []
test_acc = []
epoch = 500
loss_all = 0

# 训练部分
for epoch in range(epoch):  # 数据集级别迭代
    for step, (x_train, y_train) in enumerate(train_db):  # batch级别迭代
        with tf.GradientTape() as tape:  # 记录梯度信息
            y = tf.matmul(x_train, w1) + b1  # 神经网络乘加运算
            y = tf.nn.softmax(y)
            y_ = tf.one_hot(y_train, depth=3)  # 转换独热码
            loss = tf.reduce_mean(tf.square(y_ - y))  # 均方误差损失函数
            loss_all += loss.numpy()
        grads = tape.gradient(loss, [w1, b1])
        w1.assign_sub(lr * grads[0])  # 参数自更新
        b1.assign_sub(lr * grads[1])
    print("Epoch {}, loss: {}".format(epoch, loss_all / 4))  # 训练集尺寸120, batch尺寸32, 需循环4次
    train_loss_results.append(loss_all / 4)
    loss_all = 0  # loss_all归0, 为记录下一个epoch的loss的准备

    # 测试部分
    # total_correct为预测对的样本个数, total_number为测试的总样本数
    total_correct, total_number = 0, 0
    for x_test, y_test in test_db:
        # 使用更新后的参数进行预测
        y = tf.matmul(x_test, w1) + b1
        y = tf.nn.softmax(y)
        pred = tf.argmax(y, axis=1)  # 返回y中最大值的索引, 即预测的分类
        pred = tf.cast(pred, dtype=y_test.dtype)
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        correct = tf.reduce_sum(correct)
        total_correct += int(correct)
        total_number += x_test.shape[0]
    acc = total_correct / total_number
    test_acc.append(acc)
    print("test_acc: ", acc)
    print("---------------------------")

# 绘制loss曲线
plt.title('loss Function Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴名称
plt.ylabel('Loss')  # y轴名称
plt.plot(train_loss_results, label="$Loss$")  # 逐点画出test_acc值并连线
plt.legend()
plt.show()

# 绘制Acc曲线
plt.title('Acc Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴名称
plt.ylabel('Acc')  # y轴名称
plt.plot(test_acc, label="$Accuracy$")  # 逐点画出test_acc值并连线
plt.legend()
plt.show()
```

### 优化

##### 预备知识

- tf.where()

  > 条件语句真返回A, 条件语句假返回B
  >
  > tf.where(条件语句, 真返回A, 假返回B)

  ```python
  import tensorflow as tf
  
  a = tf.constant([1, 2, 3, 1, 1])
  b = tf.constant([0, 1, 3, 4, 5])
  c = tf.where(tf.greater(a, b), a, b) # 若a > b, 返回a对应位置的元素, 否则返回b
  print("c: ", c)
  #运行结果
  #c:  tf.Tensor([1 2 3 4 5], shape=(5,), dtype=int32)
  ```

- 返回一个[0, 1)之间的随机数

  > np.random.RandomState.rand(维度) # 维度为空, 返回标量

  ```python
  import numpy as np
  
  rdm = np.random.RandomState(seed = 1) # seed = 常数每次生成随机数相同
  a = rdm.rand() # 返回一个随机标量
  b = rdm.rand(2, 3) # 返回一个维度为2行3列的随机数矩阵
  ```

- 将两个数组按垂直方向叠加

  > np.vstack(数组1, 数组2)

  ```python
  import numpy as np
  
  a = np.array([1, 2, 3])
  b = np.array([4, 5, 6])
  c = np.vstack((a, b))
  print("c: \n", c)
  #运行结果
  #c: 
  # [[1 2 3]
  # [4 5 6]]
  ```

- np.mgrid[]

  > np.mgrid[起始值: 结束值: 步长, 起始值: 结束值: 步长, ...]

- 将x变为一维数组, '把 . 前变量拉直'

  > x.ravel()

- 使返回的间隔数值点配对

  > np.c_[数组1, 数组2, ...]

  ```python
  import numpy as np
  
  x, y = np.mgrid[1 : 3 : 1, 2 : 4 : 0.5]
  grid = np.c_[x.ravel(), y.ravel()]
  print("x: ", x)
  print("y: ", y)
  print("grad: \n", grid)
  #运行结果
  #x:  [[1. 1. 1. 1.]
  #     [2. 2. 2. 2.]]
  #y:  [[2.  2.5  3.  3.5]
  #     [2.  2.5  3.  3.5]]
  #
  #grad: 
  # [[1.  2. ]
  # [1.  2.5]
  # [1.  3. ]
  # [1.  3.5]
  # [2.  2. ]
  # [2.  2.5]
  # [2.  3. ]
  # [2.  3.5]]
  ```

- 神经网络(NN)复杂度

  - 空间复杂度

    > 层数 = 隐藏层的层数 + 1个输出层
    >
    > 总参数 = 总w + 总b

  - 时间复杂度

    > 乘加运算次数

##### 学习率

![](.\学习率.png)

- 指数衰减学习率

  > 先用较大的学习率, 快速得到最优解, 然后逐步减小学习率, 使模型再训练后期稳定
  >
  > > 指数衰减学习率 = 初始学习率 * 学习率衰减率 (当前轮数 / 多少轮衰减一次)

  ```python
  import tensorflow as tf
  
  epoch = 40 # 总轮数
  LR_BASE = 0.2 # 初始学习率
  LR_DECAY = 0.99 # 学习率衰减率
  LR_STEP= 1 # 多少轮衰减一次
  
  w = tf.Variable(tf.constant(5, dtype=tf.float32))
  
  for epoch in range(epoch):
  	lr = LR_BASE * LR_DECAY ** (epoch / LR_STEP)
      with tf.GradientTape() as tape:
          loss = tf.square(w + 1)
  	grads = tape.gradient(loss, w)
      
  	w.assign_sub(lr * grads)
  	print("After %s epoch, w is %f, loss is %f, lr is %f" % (epoch, w.numpy(), loss, lr))
  ```

##### 激活函数

- reIu函数, f(x) = max(x, 0)

	> 优点:
	>
	> 1. 解决了梯度消失
	> 2. 只需要判断输入是否大于0, 计算迅速
	> 3. 收敛速度远快于sigmod和tanh
	> 
	> 缺点:
	> 
	> 1. 输出非0均值, 收敛慢
	> 2. Dead ReIu问题: 某些神经元永远不会被激活导致参数不更新(输入负值过多, 初始化负值过多, 建议设置更小学习率, 减少负数来规避)

- Leaky ReIu函数, f(x) = max(ax, x)

	> 理论上解决了Dead ReIu的问题,并且拥有ReIu所有优点, 但是实际使用中并没有完全证明Leaky ReIu总是好于ReIu

- 对初学者建议

  - 首选reIu激活函数

  - 设置较小学习率

  - 输入特征标准化, 即让输入特征满足以0为均值, 1为标准差的正态分布

  - 初始参数中心化, 即让随机生成的参数满足以0为均值, $ \sqrt {\frac 2 {当前层输入特征个数}} $

    为标准差的正态分布
  
##### 损失函数
- 预测值(y)与已知答案(y_)的差距
  >                          ​                                               |-- mse(Mean Squared Error)
  > NN优化目标: loss最小 ----> |-- 自定义	  	
  >                          ​                                               |-- ce(Cross Entropy)

  - 均方误差mse

    > loss_mse = tf.reduce_mean(tf.square(y_ - y))

  - 自定义损失函数(预测销量为例)

    - 预测商品销量, 预测多了, 损失成本; 预测小了, 损失利润. 若 利润 != 成本 , 则mse产生的loss无法利益最大化.

    > $$
    > f(y_-, y) = 
    > \begin{cases}
    > PROFIT\ *\ (y_-\ -\ y) \ y<y_-\ 预测的少了, 损失利润 \\
    > COST\ *\ (y\ -\ y_-) \ y>=y_-\ 预测的多了, 损失成本
    > \end{cases}
    > $$
    >
    > 

    ```python
    loss_zdy = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_) * COST, (y_ - y) * PROFIT))
    ```

  - 交叉熵损失函数CE(Cross Entropy)

    - 表征两个概率分布之间的距离
    
    > $$
    > H(y_, y) = - \sum y_- * lny
    > $$
    >
    
    > tf.losses.categorical_crossentropy(y_, y)

  - softmax与交叉熵结合

    - 输出先过softmax函数, 再计算y与y_的交叉熵损失函数

    > tf.nn.softmax_cross_entropy_with_logits(y_, y)

    ```python
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 关闭log信息
    
    # softmax与交叉熵损失函数的结合
    import tensorflow as tf
    import numpy as np
    
    y_ = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]])
    y = np.array([[12, 3, 2], [3, 10, 1], [1, 2, 5], [4, 6.5, 1.2], [3, 6, 1]])
    y_pro = tf.nn.softmax(y)
    loss_ce1 = tf.losses.categorical_crossentropy(y_,y_pro)
    loss_ce2 = tf.nn.softmax_cross_entropy_with_logits(y_, y)
    
    print('分步计算的结果:\n', loss_ce1)
    print('结合计算的结果:\n', loss_ce2)
    
    # 输出的结果相同
    #分步计算的结果:
    #tf.Tensor([1.68795487e-04 1.03475622e-03 6.58839038e-02 2.58349207e+00 5.49852354e-02], shape=(5,), dtype=float64)
    #结合计算的结果:
    #tf.Tensor([1.68795487e-04 1.03475622e-03 6.58839038e-02 2.58349207e+00 5.49852354e-02], shape=(5,), dtype=float64)
    ```

##### 欠拟合和过拟合

> 欠拟合解决:
>
> > 1. 增加输入特征项
> > 2. 增加网络参数
> > 3. 减少正则化参数
>
> 过拟合解决:
>
> > 1. 清洗数据
> > 2. 增大数据集
> > 3. 采用正则化
> > 4. 增大正则化参数

- 正则化缓解过拟合

  - 正则化再损失函数张引入模型复杂度指标, 利用给w加权, 弱化了训练数据的噪声(一般不正则化b)

    - loss(y, y_): 模型中所有参数的损失函数, 如交叉熵, 均方误差
    - REGULARIZER: 用超参数REGULARIZER给出参数w在总loss中的比例, 即正则化的权重
    - w: 需要正则化的参数

    > loss = loss(y, y_) + REGULARIZER * loss(w)

  - 正则化的选择

    - L1正则化大概率会使很多参数变为0, 因此该方法可通过稀疏参数, 即减少参数的数量, 降低复杂度.

      > $$
      > \operatorname{loss}_{L{1}}(w)=\sum_{i}\left|w_{i}\right|
      > $$

    - L2正则化会使参数很接近零但不为零, 因此该方法可通过减小参数值的大小降低复杂度

      > $$
      > \operatorname{loss}_{L 2}(w)=\sum_{i}\left|w_{i}^{2}\right|
      > $$

  ```python
  # 导入所需模块
  import tensorflow as tf
  from matplotlib import pyplot as plt
  import numpy as np
  import pandas as pd
  
  # 读入数据/标签 生成x_train y_train
  df = pd.read_csv('dot.csv')
  x_data = np.array(df[['x1', 'x2']])
  y_data = np.array(df['y_c'])
  
  x_train = x_data
  y_train = y_data.reshape(-1, 1)
  
  Y_c = [['red' if y else 'blue'] for y in y_train]
  
  # 转换x的数据类型，否则后面矩阵相乘时会因数据类型问题报错
  x_train = tf.cast(x_train, tf.float32)
  y_train = tf.cast(y_train, tf.float32)
  
  # from_tensor_slices函数切分传入的张量的第一个维度，生成相应的数据集，使输入特征和标签值一一对应
  train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
  
  # 生成神经网络的参数，输入层为4个神经元，隐藏层为32个神经元，2层隐藏层，输出层为3个神经元
  # 用tf.Variable()保证参数可训练
  w1 = tf.Variable(tf.random.normal([2, 11]), dtype=tf.float32)
  b1 = tf.Variable(tf.constant(0.01, shape=[11]))
  
  w2 = tf.Variable(tf.random.normal([11, 1]), dtype=tf.float32)
  b2 = tf.Variable(tf.constant(0.01, shape=[1]))
  
  lr = 0.01  # 学习率为
  epoch = 400  # 循环轮数
  
  # 训练部分
  for epoch in range(epoch):
      for step, (x_train, y_train) in enumerate(train_db):
          with tf.GradientTape() as tape:  # 记录梯度信息
  
              h1 = tf.matmul(x_train, w1) + b1  # 记录神经网络乘加运算
              h1 = tf.nn.relu(h1)
              y = tf.matmul(h1, w2) + b2
  
              # 采用均方误差损失函数mse = mean(sum(y-out)^2)
              loss_mse = tf.reduce_mean(tf.square(y_train - y))
              # 添加l2正则化
              loss_regularization = []
              # tf.nn.l2_loss(w)=sum(w ** 2) / 2
              loss_regularization.append(tf.nn.l2_loss(w1))
              loss_regularization.append(tf.nn.l2_loss(w2))
              # 求和
              # 例：x=tf.constant(([1,1,1],[1,1,1]))
              #   tf.reduce_sum(x)
              # >>>6
              # loss_regularization = tf.reduce_sum(tf.stack(loss_regularization))
              loss_regularization = tf.reduce_sum(loss_regularization)
              loss = loss_mse + 0.03 * loss_regularization # REGULARIZER = 0.03
  
          # 计算loss对各个参数的梯度
          variables = [w1, b1, w2, b2]
          grads = tape.gradient(loss, variables)
  
          # 实现梯度更新
          # w1 = w1 - lr * w1_grad
          w1.assign_sub(lr * grads[0])
          b1.assign_sub(lr * grads[1])
          w2.assign_sub(lr * grads[2])
          b2.assign_sub(lr * grads[3])
  
      # 每200个epoch，打印loss信息
      if epoch % 20 == 0:
          print('epoch:', epoch, 'loss:', float(loss))
  
  # 预测部分
  print("*******predict*******")
  # xx在-3到3之间以步长为0.01，yy在-3到3之间以步长0.01,生成间隔数值点
  xx, yy = np.mgrid[-3:3:.1, -3:3:.1]
  # 将xx, yy拉直，并合并配对为二维张量，生成二维坐标点
  grid = np.c_[xx.ravel(), yy.ravel()]
  grid = tf.cast(grid, tf.float32)
  # 将网格坐标点喂入神经网络，进行预测，probs为输出
  probs = []
  for x_predict in grid:
      # 使用训练好的参数进行预测
      h1 = tf.matmul([x_predict], w1) + b1
      h1 = tf.nn.relu(h1)
      y = tf.matmul(h1, w2) + b2  # y为预测结果
      probs.append(y)
  
  # 取第0列给x1，取第1列给x2
  x1 = x_data[:, 0]
  x2 = x_data[:, 1]
  # probs的shape调整成xx的样子
  probs = np.array(probs).reshape(xx.shape)
  plt.scatter(x1, x2, color=np.squeeze(Y_c))
  # 把坐标xx yy和对应的值probs放入contour<[‘kɑntʊr]>函数，给probs值为0.5的所有点上色  plt点show后 显示的是红蓝点的分界线
  plt.contour(xx, yy, probs, levels=[.5])
  plt.show()
  
  # 读入红蓝点，画出分割线，包含正则化
  # 不清楚的数据，建议print出来查看 
  ```

###### 优化器

**优化器**：是引导神经网络更新参数的工具

**作用**：用来更新和计算影响模型训练和模型输出的网络参数，使其逼近或达到最优值，从而最小化(或最大化)损失函数

待优化参数w，损失函数loss, 学习率lr， 每次迭代个batch（每个batch包含2^n组数据），t表示当前batch迭代的总次数:

1. 计算t时刻损失函数关于当前参数的梯度 

$$
g_{t}=\nabla \operatorname{loss}=\frac{\partial \operatorname{loss}}{\partial\left(w_{t}\right)}
$$



2. 计算t时刻一阶动量mt和二阶动量Vt

   - 一阶动量:与梯度相关的函数

   - 二阶动量:与梯度平方相关的函数

3. 计算t时刻下降梯度:

$$
\eta_{\mathrm{t}}=l r \cdot m_{\mathrm{t}} / \sqrt{V_{\mathrm{t}}}
$$

4. 计算t+1时刻参数
   $$
   w_{\mathrm{t}+1}=w_{t}-\eta_{t}=w_{t}-l r \cdot m_{t} / \sqrt{V_{t}}
   $$

   >  不同的优化器实质上只是定义了不同的一阶动量和二阶动量公式

### SGD 随机梯度下降

-  SGD (无momentum)，常用的梯度下降法。

###### SGDM

- ( SGDM (含momentum的SGD)，在SGD基础上增加一 阶动量。

$$
m_{\mathrm{t}}=\beta \cdot m_{t-1}+(1-\beta) \cdot g_{t}\\ V_{\mathrm{t}}=1
$$

>  mt：表示各时刻梯度方向的指数滑动平均值

>  β：超参数，趋近于1，经验值为0.9

```python
m_w, m_b = 0, 0
beta = 0.9

m_w = beta * m_w + (1 - beta) * grads[0]
m_b = beta * m_b + (1 - beta) * grads[1]
w1.assign_sub(lr * m_w)
b1.assign_sub(lr * m_b)
```

###### Adagrad

- Adagrad, 在SGD基础上增加二阶动量
  $$
  m_{\mathrm{t}}=g_{\mathrm{t}}
  $$

- 二阶动量是从开始到现在梯度平方的累计和: 
  $$
  V_{t}=\sum_{\tau=1}^{t} g_{\tau}^{2}
  $$

```python
v_w, v_b = 0, 0

v_w += tf.square(grads[0])
v_b += tf.square(grads[1])
w1.assign_sub(lr * grads[0] / tf.square(v_w))
b1.assign_sub(lr * grads[1] / tf.square(v_b))
```

###### RMSProp

- SGD基础上增加二 阶动量
  $$
  m_{\mathrm{t}}=g_{\mathrm{t}}
  $$

- 二阶动量v使用指数滑动平均值计算，表征的是过去一段时间的平均值
  $$
  V_{t}=\beta \cdot V_{t-1}+(1-\beta) \cdot g_{t}^{2}
  $$

```python
v_w, v_b = 0, 0
beta = 0.9

v_w = bate * v_w + (1 - beta) * tf.square(grads[0])
v_b = bate * v_b + (1 - beta) * tf.square(grads[1])
w1.assign_sub(lr * grads[0] / tf.sqrt(v_w))
b1.assign_sub(lr * grads[0] / tf.sqrt(v_b))
```

###### Adam

- 同时结合SGDM一阶动量和RMSProp二阶动量

```python
##########################################################################
m_w, m_b = 0, 0
v_w, v_b = 0, 0
beta1, beta2 = 0.9, 0.999
delta_w, delta_b = 0, 0
global_step = 0
##########################################################################
 # adam
    m_w = beta1 * m_w + (1 - beta1) * grads[0]
    m_b = beta1 * m_b + (1 - beta1) * grads[1]
    v_w = beta2 * v_w + (1 - beta2) * tf.square(grads[0])
    v_b = beta2 * v_b + (1 - beta2) * tf.square(grads[1])

    m_w_correction = m_w / (1 - tf.pow(beta1, int(global_step)))
    m_b_correction = m_b / (1 - tf.pow(beta1, int(global_step)))
    v_w_correction = v_w / (1 - tf.pow(beta2, int(global_step)))
    v_b_correction = v_b / (1 - tf.pow(beta2, int(global_step)))

    w1.assign_sub(lr * m_w_correction / tf.sqrt(v_w_correction))
    b1.assign_sub(lr * m_b_correction / tf.sqrt(v_b_correction))
##########################################################################

```

### 使用八股搭建神经网络

##### 搭建网络sequenial

用Tensorflow API: `tf. keras`

六步法搭建神经网络

- 第一步：import相关模块，如import tensorflow as tf
- 第二步：指定输入网络的训练集和测试集，如指定训练集的输入x_train和标签y_train，测试集的输入x_test和标签y_test
- 第三步：逐层搭建网络结构，model = tf.keras.models.Sequential()
- 第四步：在model.compile()中配置训练方法，选择训练时使用的优化器、损失函数和最终评价指标
- 第五步：在model.fit()中执行训练过程，告知训练集和测试集的输入值和标签、每个batch的大小（batchsize）和数据集的迭代次数（epoch）
- 第六步：使用model.summary()打印网络结构，统计参数数目

##### Sequential()容器

Sequential()可以认为是个容器，这个容器里封装了一个神经网络结构。

```
model = tf.keras.models.Sequential ([网络结构]) #描述各层网络
```

在Sequential()中，要描述从输入层到输出层每一层的网络结构。每一层的网络结构可以是：

- 拉直层：

  ```
  tf.keras.layers.Flatten( )
  ```

  - 这一层不含计算，只是形状转换，把输入特征拉直变成一维数组

- 全连接层：

  ```
  tf.keras.layers.Dense(神经元个数，activation= "激活函数“，kernel_regularizer=哪种正则化)
  ```

  - activation (字符串给出)可选: relu、softmax、sigmoid、tanh
  - kernel_regularizer可 选: `tf.keras.regularizers.l1()`、 `tf.keras.regularizers.12()`

- 卷积层：`tf.keras.layers.Conv2D(filters =卷积核个数，kernel size=卷积核尺寸，strides=卷积步长，padding = " valid" or "same")`

- LSTM层；`tf.keras.layers.LSTM()`

##### compile配置神经网络的训练方法

告知训练时选择的优化器、损失函数和评测指标

```
model.compile(optimizer = 优化器, loss = 损失函数, metrics = ["准确率"] )
```

优化器可以是以字符串形式给出的优化器名字

Optimizer（优化器）可选:

- `'sgd'` or `tf.keras optimizers.SGD (lr=学习率,momentum=动量参数)`
- `'adagrad'` or `tf.keras.optimizers.Adagrad (lr=学习率)`
- '`adadelta'` or `tf.keras.optimizers.Adadelta (lr=学习率)`
- `'adam'` or `tf.keras.optimizers.Adam (lr=学习率，beta_ 1=0.9, beta_ 2=0.999)`

loss是（损失函数）可选:

- `'mse'` or `tf.keras losses MeanSquaredError()`
- `sparse_ categorical_crossentropy`or `tf.keras.losses.SparseCategoricalCrossentropy(from_logits =False)`
  - `from_logits`参数：有些神经网络的输出是经过了softmax等函数的概率分布，有些则不经概率分布直接输出，`from_logits`参数是在询问是否是原始输出，即没有经概率分布的输出。
  - 如果神经网络预测结果输出前经过了概率分布，这里是False
  - 如果神经网络预测结果输出前没有经过了概率分布，直接输出，这里是True

Metrics(评测指标)可选:

`'accuracy'` : y_ 和y都是数值，如y_=[1] y=[1]

`'categorical_accuracy'` : y_ 和y都是独热码(概率分布)，如y_ =[0,1,0] y=[0 256.0.695,0.048]

`'sparse_ categorical_accuracy'` : y_ 是数值，y是独热码(概率分布)，如y_ =[1] y=[0 256,0.695,0.048]

##### fit()执行训练过程

```python
model.fit (训练集的输入特征，训练集的标签，
batch_size= ，epochs=,
validation_data=(测试集的输入特征，测试集的标签),
validation_split=从训练集划分多少比例给测试集，
validation_freq =多少次epoch测试一次)
```

- `batch_ size`：每次喂入神经网络的样本数，推荐个数为：2^n
- `epochs`：要迭代多少次数据集
- `validation_data`和`validation_split`二选一
- `validation_freq`：每多少次epoch迭代使用测试集验证一次结果

##### model.summary()打印和统计

`summary()`可以打印出网络的结构和参数统计

##### 鸢尾花示例

```python
import tensorflow as tf
from sklearn import datasets
import numpy as np

x_train = datasets.load_iris().data
y_train = datasets.load_iris().target

np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
tf.random.set_seed(116)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(3, 
                          activation='softmax',
                          kernel_regularizer=tf.keras.regularizers.l2())
])

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=500, validation_split=0.2, validation_freq=20)

model.summary()
```

运行结果

```
......
 32/120 [=======>......................] - ETA: 0s - loss: 0.3005 - sparse_categorical_accuracy: 0.9375
120/120 [==============================] - 0s 83us/sample - loss: 0.3336 - sparse_categorical_accuracy: 0.9667
Epoch 499/500

 32/120 [=======>......................] - ETA: 0s - loss: 0.3630 - sparse_categorical_accuracy: 0.9688
120/120 [==============================] - 0s 125us/sample - loss: 0.3486 - sparse_categorical_accuracy: 0.9583
Epoch 500/500

 32/120 [=======>......................] - ETA: 0s - loss: 0.3122 - sparse_categorical_accuracy: 0.9688
120/120 [==============================] - 0s 142us/sample - loss: 0.3333 - sparse_categorical_accuracy: 0.9667 - val_loss: 0.4002 - val_sparse_categorical_accuracy: 1.0000
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                multiple                  15        
=================================================================
Total params: 15
Trainable params: 15
Non-trainable params: 0
_________________________________________________________________
```

鸢尾花分类神经网络，是四输入三输出的一层神经网络，参数12个w和3个b，共计15个参数，这一层是Dense全连接。

### 搭建网络class

Sequential搭建神经网络的方法，用Sequential可以搭建出上层输出就是下层输入的顺序网络结构,但是无法写出一些带有跳连的非顺序网络结构。这个时候我们可以选择用类class搭建神经网络结构。

其他步骤相同, 只是在第三步时指定自己的网络结构class

- class MyModel(Model) model=MyMode

  ```python
  class MyModel(Model):
      # 需要继承Model
  	def __init__ (self):
  		super(MyModel, self).__init__()
  		# 定义网络结构块,super继承要与类名一致
  	def cal(self, x):
  	# 调用网络结构块，实现前向传播
  		return y
  model = MyModel()
  ```

  - `__init__()`:定义所需网络结构块
  - call( )：写出前向传播

代码示例

```python
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from sklearn import datasets
import numpy as np

x_train = datasets.load_iris().data
y_train = datasets.load_iris().target

np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
tf.random.set_seed(116)

class IrisModel(Model):
    def __init__(self):
        super(IrisModel, self).__init__()
        self.d1 = Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, x):
        y = self.d1(x)
        return y

model = IrisModel()

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=500, validation_split=0.2, validation_freq=20)
model.summary()
```

运行结果

```python
......
 32/120 [=======>......................] - ETA: 0s - loss: 0.3630 - sparse_categorical_accuracy: 0.9688
120/120 [==============================] - 0s 108us/sample - loss: 0.3486 - sparse_categorical_accuracy: 0.9583
Epoch 500/500

 32/120 [=======>......................] - ETA: 0s - loss: 0.3122 - sparse_categorical_accuracy: 0.9688
120/120 [==============================] - 0s 158us/sample - loss: 0.3333 - sparse_categorical_accuracy: 0.9667 - val_loss: 0.4002 - val_sparse_categorical_accuracy: 1.0000
Model: "iris_model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                multiple                  15        
=================================================================
Total params: 15
Trainable params: 15
Non-trainable params: 0
_________________________________________________________________
```

### 模型使用

前向传播执行应用

```python
predict(输入特征，batch_size=整数)
#  返回前向传播计算结果
```

实现步骤

```python
# 复现模型,(前向传播)
model = tf.keras.models.Sequential([
	tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128activation='relu'),
	tf.keras.layers.Dense(10，activation='softmax')])

# 加载参数
model.load_weights(model_save_path)

# 预测结果
result = model.predict(x_ predict)
```

❗知识点补充：

- np.newaxis：在`np.newaxis`所在的位置，增加一个维度

```python
# -*- coding: utf-8 -*-
"""
tf.newaxis 和 numpy newaxis
"""
import numpy as np
import tensorflow as tf

feature = np.array([[1, 2, 3],
                        [2, 4, 6]])
center = np.array([[1, 1, 1],
                   [0, 0, 0]])

print("原始数组大小：")
print(feature.shape)
print(center.shape)

np_feature_1 = feature[:, :, np.newaxis]  # 在末尾增加一个维度
np_feature_2 = feature[:, np.newaxis] # 在中间增加一个维度
np_center = center[np.newaxis, :] # 在首部增加一个维度

print("添加 np.newaxis 后数组大小：")
print(np_feature_1.shape)
print(np_feature_1)
print('-----')
print(np_feature_2.shape)
print(np_feature_2)
print('-----')
print(np_center.shape)
print(np_center)
```

运行结果

```python
原始数组大小：
(2, 3)
(2, 3)
添加 np.newaxis 后数组大小：
(2, 3, 1)
[[[1]
  [2]
  [3]]

 [[2]
  [4]
  [6]]]
-----
(2, 1, 3)
[[[1 2 3]]

 [[2 4 6]]]
-----
(1, 2, 3)
[[[1 1 1]
  [0 0 0]]]
```

在tensorflow中有有`tf.newaxis`用法相同

```python
# -*- coding: utf-8 -*-
"""
tf.newaxis 和 numpy newaxis
"""
import numpy as np
import tensorflow as tf

feature = np.array([[1, 2, 3],
                        [2, 4, 6]])
center = np.array([[1, 1, 1],
                   [0, 0, 0]])

print("原始数组大小：")
print(feature.shape)
print(center.shape)

tf_feature_1 = feature[:, :, tf.newaxis]  # 在末尾增加一个维度
tf_feature_2 = feature[:, tf.newaxis] # 在中间增加一个维度
tf_center = center[tf.newaxis, :] # 在首部增加一个维度

print("添加 np.newaxis 后数组大小：")
print(tf_feature_1.shape)

print('-----')
print(tf_feature_2.shape)

print('-----')
print(tf_center.shape)
```

运行结果

```python
原始数组大小：
(2, 3)
(2, 3)
添加 np.newaxis 后数组大小：
(2, 3, 1)
-----
(2, 1, 3)
-----
(1, 2, 3)
```

对于手写图片运用识别模型进行判断

```python
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

model_save_path = './checkpoint/mnist.ckpt'
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.load_weights(model_save_path)
preNum = int(input("input the number of test pictures:"))

for i in range(preNum):
    image_path = input("the path of test picture:")
    img = Image.open(image_path)

    image = plt.imread(image_path)
    plt.set_cmap('gray')
    plt.imshow(image)

    img = img.resize((28, 28), Image.ANTIALIAS)
    img_arr = np.array(img.convert('L'))

    # 将输入图片变为只有黑色和白色的高对比图片
    for i in range(28):
        for j in range(28):
            if img_arr[i][j] < 200:  # 小于200的变为纯黑色
                img_arr[i][j] = 255
            else:
                img_arr[i][j] = 0  # 其余变为纯白色
	
	# 由于神经网络训练时都是按照batch输入
    # 为了满足神经网络输入特征的shape(图片总数，宽，高）
    # 所以要将28行28列的数据[28,28]二维数据---变为--->一个28行28列的数据[1,28,28]三维数据
    img_arr = img_arr / 255.0
    x_predict = img_arr[tf.newaxis, ...]  # 插入一个维度
    result = model.predict(x_predict)
    pred = tf.argmax(result, axis=1)

    print('\n')
    tf.print(pred)

    plt.pause(1)  # 相当于plt.show()，但是只显示1秒
    plt.close()
```
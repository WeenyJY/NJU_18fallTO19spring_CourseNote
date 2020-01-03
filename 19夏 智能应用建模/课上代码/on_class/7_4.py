import tensorflow as tf
import numpy as np

'''
#激活函数实现非线性化
w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1),name='w1')
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1),name='w2')
x=tf.placeholder(tf.float32,shape=(None,2),name='input')

biases1=-0.5
biases2=0.1
a=tf.nn.relu(tf.matmul(x,w1)+biases1)
y=tf.nn.relu(tf.matmul(a,w2)+biases2)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(y,feed_dict={x:[[0.7,0.9],[0.1,0.4],[0.5,0.8]]}))
'''


'''
#交叉熵H(p,q)：p为正确值，q为预测值  h=np.array([[1,0,0],[0.5,0.4,0.1]])
def cross_entropy(p,q):
    return -p.dot(np.log(q))
print(cross_entropy(np.array([1,0,0]),np.array([0.5,0.4,0.1])))
'''


'''
from numpy.random import RandomState
rdm=RandomState(1)  #产生0-1之间的一个随机数
dataset_size=128
X=rdm.rand(dataset_size,2)
Y=[[x1 + x2 + rdm.rand()/10.0 - 0.05] for (x1, x2) in X]

batch_size=8
x=tf.placeholder(tf.float32,shape=(None,2),name='x-input')
y_=tf.placeholder(tf.float32,shape=(None,1),name='y-input')

w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1)) 
y = tf.matmul(x, w1)

loss_less=10
loss_more=1
loss=tf.reduce_mean(tf.where(tf.greater(y,y_),(y-y_)*loss_more,(y_-y)*loss_less))

train_step=tf.train.AdamOptimizer(0.001).minimize(loss)

with tf.Session() as sess: 
    init_op = tf.global_variables_initializer() 
    sess.run(init_op) 
    STEPS = 5000 
    for i in range(STEPS): 
        start = (i * batch_size) % dataset_size 
        end = min(start + batch_size, dataset_size) 
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]}) 
    print(sess.run(w1))
'''


'''
#滑动平均模型的TensorFlow实现

# 定义了一个变量用于计算滑动平均，其初始值为 0 
v1=tf.Variable(0,dtype=tf.float32)     
#step 变量模拟神经网络中迭代的轮数，用于动态控制衰减率 
step= tf.Variable(0, trainable=False)

# 定义一个滑动平均的类，初始化时给定了衰减率（ 0.99 ）和控制衰减率的变量 
ema = tf.train.ExponentialMovingAverage(0.99, step) 
#定义一个变量滑动平均的操作
maintain_averanges_op=ema.apply([v1])

with tf.Session() as sess: 
    # 初始化所有变量 
    init_op = tf.global_variables_initializer() 
    sess.run(init_op) 
    
    # 通过 ema.average(v1) 获取滑动平均后变量的取值 step 
    print(sess.run([v1, ema.average(v1)]))       # 输出[0.0 , 0.0]

    sess.run(tf.assign(v1,5))    #更新变量v1的值到5
    #更新 v1 的滑动平均值，衰减率为 min{0.99, (1+step)/(10+step)=0.1}=0.1 
    sess.run(maintain_averanges_op)     #v1 的滑动平均更新为 0.1 * 0+0.9 * 5 = 4.5 
    print(sess.run([v1,ema.average(v1)]))

    sess.run(tf.assign(step,10000))   #  更新 step 的值为 10000
    sess.run(tf.assign(v1,10))            #更新v1的值为10
    #衰减率为 min{0.99, (1+step)/(10+step)=0.999}=0.99 
    # v1 的滑动平均更新为 0.99 * 4.5+0.01 * 10 = 4.555 
    sess.run(maintain_averanges_op)
    print(sess.run([v1,ema.average(v1)]))
'''


'''
#去掉激活函数的神经网络
from tensorflow.examples.tutorials.mnist import input_data
input_node=784     #输入层节点数
output_node=10      #输出层节点数

layer1_node=500     #隐藏层节点数
batch_size=100           #一个训练batch中的训练数据个数
training_steps=1000    #训练轮数

#辅助函数计算神经网络的前向传播结果
def inference(input_tensor, weights1, biases1, weights2, biases2): 
    # 计算隐藏层的前向传播结果，通过加入隐藏层实现了多层网络结构 
    # 这里使用了 ReLU 激活函数，通过 ReLU 激活函数实现了去线性化 
    #layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1) 
    layer1=tf.matmul(input_tensor,weights1)+biases1
    # 计算输出层的前向传播结果，这里不需要加入激活函数 
    return tf.matmul(layer1, weights2) + biases2


def train(mnist): 
    x = tf.placeholder(tf.float32, [None, input_node], name='x-input') 
    y_ = tf.placeholder(tf.float32, [None, output_node], name='y-input')
    # 生成隐藏层的参数 
    weights1 = tf.Variable(tf.truncated_normal([input_node, layer1_node], stddev=0.1)) 
    biases1 = tf.Variable(tf.constant(0.1, shape=[layer1_node]))
    # 生成输出层的参数 
    weights2 = tf.Variable(tf.truncated_normal([layer1_node, output_node], stddev=0.1)) 
    biases2 = tf.Variable(tf.constant(0.1, shape=[output_node]))
    y = inference(x, weights1, biases1, weights2, biases2)

    # 计算交叉熵，这里交叉熵作为刻画预测值和真实值之间差距的损失函数 
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))

    # 计算在当前 batch 中所有样例的交叉熵平均值 
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # 使用梯度下降算法来优化损失函数 
    train_step = tf.train.GradientDescentOptimizer(0.8).minimize(cross_entropy_mean)
    #tf.argmax 的第二个参数 "1" 表示选取最大值的操作仅在第一个维度进行，也就是说， 
    # 只在每一行选取最大值对应的下标。于是得到的结果是一个长度为 batch 的一维数组， 
    # 这个一维数组中的值就表示了每一个样例对应的数字识别结果 
    # tf.equal 判断两个张量的每一维是否相等，如果相等返回 True ，否则返回 False 
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # 先将布尔型的数值转换为实数型，然后计算平均值 
    # 这个平均值就是模型在这一组数据上的正确率 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess: 
        tf.global_variables_initializer().run()
    #  准备验证数据。在训练过程中通过验证数据来大致判断停止的条件和评判训练的效果 
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

    # 迭代地训练神经网络 
        for i in range(training_steps): 
        # 产生这一轮使用的一个 batch 的训练数据，并运行训练过程 
            xs, ys = mnist.train.next_batch(batch_size) 
            sess.run(train_step, feed_dict={x: xs, y_: ys})
        #  每 100 轮输出一次在验证数据集上的测试结果 
            if i % 100 == 0: 
                validate_acc = sess.run(accuracy, feed_dict=validate_feed) 
                print("After %d training steps, validation accuracy is %g" % (i, validate_acc))

    # 准备测试数据。这部分数据在训练时是不可见的，训练后它们用于对模型优劣做最后的评价 
        test_feed ={x: mnist.test.images, y_: mnist.test.labels}
    # 在测试数据上检验神经网络模型的最终准确率 
        test_acc = sess.run(accuracy, feed_dict=test_feed) 
        print("After %d training steps, test accuracy is %g" % (training_steps, test_acc))
 
def main(argv=None):
    mnist=input_data.read_data_sets(r'C:/Users/Jaqen/Desktop/mnist',one_hot=True)
    train(mnist)

if __name__=='__main__':
    main()                  # tf.app.run()
'''


'''
#加入指数衰减学习率
from tensorflow.examples.tutorials.mnist import input_data
input_node=784     #输入层节点数
output_node=10      #输出层节点数

layer1_node=500     #隐藏层节点数
batch_size=100           #一个训练batch中的训练数据个数
training_steps=1000    #训练轮数

learning_rate_base=0.8
learning_rate_decay=0.99

#辅助函数计算神经网络的前向传播结果
def inference(input_tensor, weights1, biases1, weights2, biases2): 
    # 计算隐藏层的前向传播结果，通过加入隐藏层实现了多层网络结构 
    # 这里使用了 ReLU 激活函数，通过 ReLU 激活函数实现了去线性化 
    layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1) 
    # 计算输出层的前向传播结果，这里不需要加入激活函数 
    return tf.matmul(layer1, weights2) + biases2


def train(mnist): 
    x = tf.placeholder(tf.float32, [None, input_node], name='x-input') 
    y_ = tf.placeholder(tf.float32, [None, output_node], name='y-input')
    # 生成隐藏层的参数 
    weights1 = tf.Variable(tf.truncated_normal([input_node, layer1_node], stddev=0.1)) 
    biases1 = tf.Variable(tf.constant(0.1, shape=[layer1_node]))
    # 生成输出层的参数 
    weights2 = tf.Variable(tf.truncated_normal([layer1_node, output_node], stddev=0.1)) 
    biases2 = tf.Variable(tf.constant(0.1, shape=[output_node]))
    y = inference(x, weights1, biases1, weights2, biases2)

    global_step=tf.Variable(0,trainable=False)
    learning_rate=tf.train.exponential_decay(learning_rate_base,global_step,mnist.train.num_examples/batch_size,learning_rate_decay)
    
    # 计算交叉熵，这里交叉熵作为刻画预测值和真实值之间差距的损失函数 
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))

    # 计算在当前 batch 中所有样例的交叉熵平均值 
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # 使用梯度下降算法来优化损失函数 
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_mean,global_step=global_step)
    #tf.argmax 的第二个参数 "1" 表示选取最大值的操作仅在第一个维度进行，也就是说， 
    # 只在每一行选取最大值对应的下标。于是得到的结果是一个长度为 batch 的一维数组， 
    # 这个一维数组中的值就表示了每一个样例对应的数字识别结果 
    # tf.equal 判断两个张量的每一维是否相等，如果相等返回 True ，否则返回 False 
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # 先将布尔型的数值转换为实数型，然后计算平均值 
    # 这个平均值就是模型在这一组数据上的正确率 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess: 
        tf.global_variables_initializer().run()
    #  准备验证数据。在训练过程中通过验证数据来大致判断停止的条件和评判训练的效果 
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

    # 迭代地训练神经网络 
        for i in range(training_steps): 
        # 产生这一轮使用的一个 batch 的训练数据，并运行训练过程 
            xs, ys = mnist.train.next_batch(batch_size) 
            sess.run(train_step, feed_dict={x: xs, y_: ys})
        #  每 100 轮输出一次在验证数据集上的测试结果 
            if i % 100 == 0: 
                validate_acc = sess.run(accuracy, feed_dict=validate_feed) 
                print("After %d training steps, validation accuracy is %g" % (i, validate_acc))

    # 准备测试数据。这部分数据在训练时是不可见的，训练后它们用于对模型优劣做最后的评价 
        test_feed ={x: mnist.test.images, y_: mnist.test.labels}
    # 在测试数据上检验神经网络模型的最终准确率 
        test_acc = sess.run(accuracy, feed_dict=test_feed) 
        print("After %d training steps, test accuracy is %g" % (training_steps, test_acc))
 
def main(argv=None):
    mnist=input_data.read_data_sets(r'C:/Users/Jaqen/Desktop/mnist',one_hot=True)
    train(mnist)

if __name__=='__main__':
    main()
'''

'''
#再加入正则化
from tensorflow.examples.tutorials.mnist import input_data
input_node=784     #输入层节点数
output_node=10      #输出层节点数

layer1_node=500     #隐藏层节点数
batch_size=100           #一个训练batch中的训练数据个数
training_steps=1000    #训练轮数

learning_rate_base=0.8
learning_rate_decay=0.99

regularization_rate=0.0001

#辅助函数计算神经网络的前向传播结果
def inference(input_tensor, weights1, biases1, weights2, biases2): 
    # 计算隐藏层的前向传播结果，通过加入隐藏层实现了多层网络结构 
    # 这里使用了 ReLU 激活函数，通过 ReLU 激活函数实现了去线性化 
    layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1) 
    # 计算输出层的前向传播结果，这里不需要加入激活函数 
    return tf.matmul(layer1, weights2) + biases2


def train(mnist): 
    x = tf.placeholder(tf.float32, [None, input_node], name='x-input') 
    y_ = tf.placeholder(tf.float32, [None, output_node], name='y-input')
    # 生成隐藏层的参数 
    weights1 = tf.Variable(tf.truncated_normal([input_node, layer1_node], stddev=0.1)) 
    biases1 = tf.Variable(tf.constant(0.1, shape=[layer1_node]))
    # 生成输出层的参数 
    weights2 = tf.Variable(tf.truncated_normal([layer1_node, output_node], stddev=0.1)) 
    biases2 = tf.Variable(tf.constant(0.1, shape=[output_node]))
    y = inference(x, weights1, biases1, weights2, biases2)

    global_step=tf.Variable(0,trainable=False)
    learning_rate=tf.train.exponential_decay(learning_rate_base,global_step,mnist.train.num_examples/batch_size,learning_rate_decay)
    
    # 计算交叉熵，这里交叉熵作为刻画预测值和真实值之间差距的损失函数 
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # 计算在当前 batch 中所有样例的交叉熵平均值 
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    #定义l2正则化损失
    regularizer=tf.contrib.layers.l2_regularizer(regularization_rate)
    #计算模型的正则化损失
    regularization=regularizer(weights1)+regularizer(weights2)
    loss=cross_entropy_mean+regularization

    # 使用梯度下降算法来优化损失函数 
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    #tf.argmax 的第二个参数 "1" 表示选取最大值的操作仅在第一个维度进行，也就是说， 
    # 只在每一行选取最大值对应的下标。于是得到的结果是一个长度为 batch 的一维数组， 
    # 这个一维数组中的值就表示了每一个样例对应的数字识别结果 
    # tf.equal 判断两个张量的每一维是否相等，如果相等返回 True ，否则返回 False 
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # 先将布尔型的数值转换为实数型，然后计算平均值 
    # 这个平均值就是模型在这一组数据上的正确率 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess: 
        tf.global_variables_initializer().run()
    #  准备验证数据。在训练过程中通过验证数据来大致判断停止的条件和评判训练的效果 
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

    # 迭代地训练神经网络 
        for i in range(training_steps): 
        # 产生这一轮使用的一个 batch 的训练数据，并运行训练过程 
            xs, ys = mnist.train.next_batch(batch_size) 
            sess.run(train_step, feed_dict={x: xs, y_: ys})
        #  每 100 轮输出一次在验证数据集上的测试结果 
            if i % 100 == 0: 
                validate_acc = sess.run(accuracy, feed_dict=validate_feed) 
                print("After %d training steps, validation accuracy is %g" % (i, validate_acc))

    # 准备测试数据。这部分数据在训练时是不可见的，训练后它们用于对模型优劣做最后的评价 
        test_feed ={x: mnist.test.images, y_: mnist.test.labels}
    # 在测试数据上检验神经网络模型的最终准确率 
        test_acc = sess.run(accuracy, feed_dict=test_feed) 
        print("After %d training steps, test accuracy is %g" % (training_steps, test_acc))
 
def main(argv=None):
    mnist=input_data.read_data_sets(r'C:/Users/Jaqen/Desktop/mnist',one_hot=True)
    train(mnist)

if __name__=='__main__':
    main()
'''


'''
#上下文管理器
v1 = tf.get_variable("v", [1]) 
print(v1.name)  
# 输出 v:0 
with tf.variable_scope("foo"): 
    v2 = tf.get_variable("v", [1]) 
    print(v2.name)   
# 输出 foo/v:0 
with tf.variable_scope("foo"): 
    with tf.variable_scope("bar"): 
        v3 = tf.get_variable("v", [1]) 
        print(v3.name)   # 输出 foo/bar/v:0 
    v4 = tf.get_variable("v1", [1]) 
    print(v4.name)   # 输出 foo/v1:0
'''

'''
#加入上下文管理器的inference函数
from tensorflow.examples.tutorials.mnist import input_data
input_node=784     #输入层节点数
output_node=10      #输出层节点数

layer1_node=500     #隐藏层节点数
batch_size=100           #一个训练batch中的训练数据个数
training_steps=1000    #训练轮数

learning_rate_base=0.8
learning_rate_decay=0.99

regularization_rate=0.0001

#辅助函数计算神经网络的前向传播结果
def inference(input_tensor): 
    with tf.variable_scope('layer1'): 
        weights = tf.get_variable("weights", [input_node, layer1_node], initializer=tf.truncated_normal_initializer(stddev=0.1)) 
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularization_rate)(weights)) 
        biases = tf.get_variable("biases", [layer1_node], initializer=tf.constant_initializer(0.0)) 
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
    with tf.variable_scope('layer2'): 
        weights = tf.get_variable("weights", [layer1_node, output_node], initializer=tf.truncated_normal_initializer(stddev=0.1)) 
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularization_rate)(weights)) 
        biases = tf.get_variable("biases", [output_node], initializer=tf.constant_initializer(0.0)) 
        layer2 = tf.matmul(layer1, weights) + biases 
    return layer2


def train(mnist): 
    x = tf.placeholder(tf.float32, [None, input_node], name='x-input') 
    y_ = tf.placeholder(tf.float32, [None, output_node], name='y-input')
    
    global_step=tf.Variable(0,trainable=False)
    learning_rate=tf.train.exponential_decay(learning_rate_base,global_step,mnist.train.num_examples/batch_size,learning_rate_decay)
    
    y=inference(x)
    # 计算交叉熵，这里交叉熵作为刻画预测值和真实值之间差距的损失函数 
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # 计算在当前 batch 中所有样例的交叉熵平均值 
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regularization=tf.add_n(tf.get_collection('losses'))
    loss=cross_entropy_mean+regularization

    # 使用梯度下降算法来优化损失函数 
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    #tf.argmax 的第二个参数 "1" 表示选取最大值的操作仅在第一个维度进行，也就是说， 
    # 只在每一行选取最大值对应的下标。于是得到的结果是一个长度为 batch 的一维数组， 
    # 这个一维数组中的值就表示了每一个样例对应的数字识别结果 
    # tf.equal 判断两个张量的每一维是否相等，如果相等返回 True ，否则返回 False 
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # 先将布尔型的数值转换为实数型，然后计算平均值 
    # 这个平均值就是模型在这一组数据上的正确率 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess: 
        tf.global_variables_initializer().run()
    #  准备验证数据。在训练过程中通过验证数据来大致判断停止的条件和评判训练的效果 
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

    # 迭代地训练神经网络 
        for i in range(training_steps): 
        # 产生这一轮使用的一个 batch 的训练数据，并运行训练过程 
            xs, ys = mnist.train.next_batch(batch_size) 
            sess.run(train_step, feed_dict={x: xs, y_: ys})
        #  每 100 轮输出一次在验证数据集上的测试结果 
            if i % 100 == 0: 
                validate_acc = sess.run(accuracy, feed_dict=validate_feed) 
                print("After %d training steps, validation accuracy is %g" % (i, validate_acc))

    # 准备测试数据。这部分数据在训练时是不可见的，训练后它们用于对模型优劣做最后的评价 
        test_feed ={x: mnist.test.images, y_: mnist.test.labels}
    # 在测试数据上检验神经网络模型的最终准确率 
        test_acc = sess.run(accuracy, feed_dict=test_feed) 
        print("After %d training steps, test accuracy is %g" % (training_steps, test_acc))
 
def main(argv=None):
    mnist=input_data.read_data_sets(r'C:/Users/Jaqen/Desktop/mnist',one_hot=True)
    train(mnist)

if __name__=='__main__':
    main()
'''

'''
#模型持久化小样例
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1") 
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2") 
result = v1 + v2
init_op = tf.global_variables_initializer()
# 声明 tf.train.Saver 类用于保存模型 
saver = tf.train.Saver()
with tf.Session() as sess: 
    sess.run(init_op) 
    # 将模型保存到 /path/model/model.ckpt 文件 
    saver.save(sess, "C:/Users/Jaqen/Desktop/path/model/model.ckpt")
    #用restore函数加载已经保存的模型
    saver.restore(sess,'C:/Users/Jaqen/Desktop/path/model/model.ckpt')
    print(sess.run(result))
'''
'''
#直接加载已经持久化的图
saver = tf.train.import_meta_graph("C:/Users/Jaqen/Desktop/path/model/model.ckpt.meta")
with tf.Session() as sess: 
    saver.restore(sess, "C:/Users/Jaqen/Desktop/path/model/model.ckpt") 
    # 通过张量的名称来获取张量，输出 [3.] 
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))
'''




























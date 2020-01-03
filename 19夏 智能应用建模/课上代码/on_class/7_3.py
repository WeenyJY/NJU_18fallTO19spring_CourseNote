'''
@Descripttion: 
@version: 
@Author: Zhou Renzhe
@Date: 2019-07-02 18:56:48
@LastEditTime: 2019-07-03 16:28:54
'''

import tensorflow as tf

'''
a=tf.constant([1,2],name='a')
b=tf.constant([1,3],name='b')
sess=tf.Session()
result=a+b
sess.run(result)
print(result)
sess.close()
'''


'''
#创建计算图
with tf.name_scope("input1"):
    input1=tf.constant([1.0,2.0,3.0],name='input1')
with tf.name_scope("input2"):
    input2=tf.Variable(tf.random_uniform([3]),name='input2')
output=tf.add_n([input1,input2],name='add')
writer=tf.summary.FileWriter('C:/Users/Jaqen/Desktop',tf.get_default_graph())   
writer.close()
'''


'''
#计算图的另一种创建方式
g1=tf.Graph()
with g1.as_default():
    v=tf.get_variable("v",shape=[1],initializer=tf.zeros_initializer)

g2=tf.Graph()
with g2.as_default():
    v=tf.get_variable("v",shape=[1],initializer=tf.ones_initializer)

# 在计算图g1中读取变量“v”的取值 
with tf.Session(graph=g1) as sess: 
    tf.initialize_all_variables().run() 
    with tf.variable_scope("", reuse=True): 
        # 在计算图g1中，变量“v”的取值应该为0，所以下面这行会输出[0.] 
        print(sess.run(tf.get_variable("v"))) 
# 在计算图g2中读取变量“v”的取值 
with tf.Session(graph=g2) as sess: 
    tf.initialize_all_variables().run() 
    with tf.variable_scope("", reuse=True): 
        # 在计算图g2中，变量“v”的取值应该为1，所以下面这行会输出[1.] 
        print(sess.run(tf.get_variable("v"))) 
writer = tf.summary.FileWriter("C:/Users/Jaqen/Desktop/path/to/log", g1) 
writer = tf.summary.FileWriter("C:/Users/Jaqen/Desktop/path/to/log", g2) 
writer.close()
'''


'''
#前向传播算法的实例
w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1),name='w1')
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1),name='w2')
x=tf.constant([[0.7,0.9]],name='x')

a=tf.matmul(x,w1,name='a')
y=tf.matmul(a,w2,name='y')

sess=tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(w1))
print(sess.run(y))

writer = tf.summary.FileWriter("C:/Users/Jaqen/Desktop/path", tf.get_default_graph()) 
writer.close()
'''

'''
#placeholder机制
w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1),name='w1')
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1),name='w2')
x=tf.placeholder(tf.float32,shape=(1,2),name='input')

a=tf.matmul(x,w1,name='a')
y=tf.matmul(a,w2,name='y')

sess=tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(y,feed_dict={x:[[0.7,0.9],[0.1,0.4],[0.5,0.8]]})

y=tf.sigmoid(y)
cross_entropy=-tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0))+(1-y_)*tf.log(tf.clip_by_value(1-y,1e-10,1.0)))
learning_rate=0.001
train_step=tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

sess.run(train_step)
sess.close()
'''


#神经网络实例
from numpy.random import RandomState

rdm=RandomState(1)
dataset_size=128
X=rdm.rand(dataset_size,2)

Y=[[int(x1+x2<1)] for (x1,x2) in X]
batch_size=8

w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1),name='w1')
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1),name='w2')
x=tf.placeholder(tf.float32,shape=(None,2),name='x-input')
y_=tf.placeholder(tf.float32,shape=(None,1),name='y-input')

a=tf.matmul(x,w1)
y=tf.matmul(a,w2)

y=tf.sigmoid(y)
cross_entropy=-tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0))+(1-y_)*tf.log(tf.clip_by_value(1-y,1e-10,1.0)))
learning_rate=0.001
train_step=tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))

    steps=5000     #训练轮数
    for i in range(steps):
        #每次选batch_size个样本训练
        start=(i*batch_size)%dataset_size
        end=min(start+batch_size,dataset_size)

        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        if i%500==0:
            total_cross_entropy=sess.run(cross_entropy,feed_dict={x:X,y_:Y})
            print('After %d training steps,cross entropy on all data is %g'%(i,total_cross_entropy))
    
    print(sess.run(w1))
    print(sess.run(w2))






































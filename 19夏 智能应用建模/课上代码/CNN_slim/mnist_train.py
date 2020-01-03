import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import mnist_inference

batch_size=100
learning_rate_base=0.01
learning_rate_decay=0.99
training_steps=2000
moving_average_decay=0.99
regularization_rate=0.0001
LENET=False
model_save_path='C:/Users/Jaqen/Desktop/path/mnist_model'
model_name='minist_model.ckpt'


def train(mnist): 
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [batch_size, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
        
    with tf.name_scope('input_reshape'): 
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1]) 
        tf.summary.image('input', image_shaped_input, 10)
    
    global_step=tf.Variable(0,trainable=False)
    regularizer=tf.contrib.layers.l2_regularizer(regularization_rate)
    y=mnist_inference.inference_slim(x,True,regularizer)
    tf.summary.histogram('inference_output', y)

    with tf.name_scope('loss_function'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        if LENET: 
            loss = cross_entropy_mean 
            learning_rate = 0.01 
        else: # 计算模型的正则化损失 
            regularization = tf.add_n(tf.get_collection('losses')) 
            # 总损失等于交叉熵损失和正则化损失的和 
            loss = cross_entropy_mean + regularization
            learning_rate = tf.train.exponential_decay( learning_rate_base, global_step, mnist.train.num_examples / batch_size, learning_rate_decay)
            # 过完所有的训练数据需要的迭代次数 
        
        #regularization=tf.add_n(tf.get_collection('losses'))
        #loss=cross_entropy_mean+regularization
        tf.summary.scalar('loss', loss) 
   
    with tf.name_scope('train_step'):
        
        learning_rate=tf.train.exponential_decay(learning_rate_base,global_step,mnist.train.num_examples/batch_size,learning_rate_decay)
        # 使用梯度下降算法来优化损失函数 
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) 
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        saver=tf.train.Saver()
        merged = tf.summary.merge_all() # 整理所有的日志生成操作 
        
        with tf.Session() as sess: 
            summary_writer = tf.summary.FileWriter("C:/Users/Jaqen/Desktop/path", sess.graph) 
            tf.global_variables_initializer().run()
    #  准备验证数据。在训练过程中通过验证数据来大致判断停止的条件和评判训练的效果 
            validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        
            for i in range(training_steps):
                xs,ys=mnist.train.next_batch(batch_size)
                reshaped_xs = np.reshape(xs, (batch_size, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS)) 
                summary, _, loss_value, step = sess.run([merged, train_step, loss, global_step], feed_dict={x: reshaped_xs, y_: ys}) 
                summary_writer.add_summary(summary, i) 
                if i%2!=0:
                    print('After %d training steps, loss on training batch is %g'%(step,loss_value))
                    saver.save(sess,os.path.join(model_save_path,model_name),global_step=global_step)
            summary_writer.close()
    writer = tf.summary.FileWriter("C:/Users/Jaqen/Desktop/path", tf.get_default_graph()) 
    writer.close()


def main(argv=None):
    mnist=input_data.read_data_sets('C:/Users/Jaqen/Desktop/mnist',one_hot=True)
    train(mnist)

if __name__=='__main__':
    main()
















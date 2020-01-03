import time 
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
# 加载 mnist_inference.py 和 mnist_train.py 中定义的常量和函数 
import mnist_inference 
import mnist_train
# 每 1 秒加载一次最新的模型，并在测试数据上测试最新模型的正确率 
eval_interval_secs = 1

def evaluate(mnist): 
    with tf.Graph().as_default() as g: 
        # 定义输入输出的格式 
        x = tf.placeholder(tf.float32, [None, mnist_inference.input_node], name='x-input') 
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.output_node], name='y-input') 
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        # 直接通过调用封装好的函数来计算前向传播的结果 
        y = mnist_inference.inference(x)
        # 使用前向传播的结果计算正确率 
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) 
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        saver = tf.train.Saver()

        while True: 
            with tf.Session() as sess: 
                #tf.train.get_checkpoint_state 函数会通过 checkpoint 文件自动找到目录中最 新模型的文件名 
                ckpt = tf.train.get_checkpoint_state(mnist_train.model_save_path) 
                if ckpt and ckpt.model_checkpoint_path: 
                    # 加载模型 
                    saver.restore(sess, ckpt.model_checkpoint_path) 
                    # 通过文件名得到模型保存时迭代的轮数 
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1] 
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed) 
                    print("After %s training steps, validation accuarcy = %g" % (global_step, accuracy_score)) 
                else: 
                    print('No checkpoint file found') 
                    return 
            time.sleep(eval_interval_secs) 

def main(argv=None): 
    mnist = input_data.read_data_sets("C:/Users/Jaqen/Desktop/mnist", one_hot=True) 
    evaluate(mnist)
if __name__ == '__main__':
    main()





    
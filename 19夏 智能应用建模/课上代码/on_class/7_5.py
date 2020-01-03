import tensorflow as tf

with tf.variable_scope('layer1-conv1'): 
    conv1_weights = tf.get_variable("weight", [5, 5, 1, 6], initializer=tf.truncated_normal_initializer(stddev=0.1)) 
    conv1_biases = tf.get_variable("biases", [6], initializer=tf.constant_initializer(0.0))
    conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='VALID') 
    relu1= tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

with tf.name_scope('layer2-pool1'): 
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.variable_scope('layer3-conv2'): 
    conv2_weights = tf.get_variable("weight", [5, 5, 6, 16], initializer=tf.truncated_normal_initializer(stddev=0.1)) 
    conv2_biases = tf.get_variable("bias", [16], initializer=tf.constant_initializer(0.0))
    conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding="SAME") 
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

with tf.name_scope('layer4-pool2'): 
    pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=”SAME")

# 将第四层池化层的输出转化为第五层全连接层的输入格式 
# #pool2.get_shape 函数可以得到第四层输出矩阵的维度而不需要手工计算 
pool_shape = pool2.get_shape().as_list() 
# 将这个 5x5x16 的矩阵拉直成一个向量 
nodes = pool_shape[1] * pool_shape[2] * pool_shape[3] 
#pool_shape[0] 代表一个 batch 中数据的个数 
reshaped = tf.reshape(pool2, [pool_shape[0], nodes])
with tf.variable_scope('layer5-fc1'): 
    fc1_weights = tf.get_variable("weight", [nodes, 120], initializer=tf.truncated_normal_initializer(stddev=0.1)) 
    fc1_biases = tf.get_variable("bias", [120], initializer=tf.constant_initializer(0.1)) 
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)

with tf.variable_scope('layer6-fc2'): 
    fc2_weights = tf.get_variable("weight", [120, 84], initializer=tf.truncated_normal_initializer(stddev=0.1)) 
    fc2_biases = tf.get_variable("bias", [84],  initializer=tf.constant_initializer(0.1)) 
    fc2 = tf.matmul(fc1, fc2_weights) + fc2_biases


















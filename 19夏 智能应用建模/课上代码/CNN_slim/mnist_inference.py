import tensorflow as tf

# 配置神经网络的参数 
INPUT_NODE = 784 
OUTPUT_NODE = 10
IMAGE_SIZE = 28 
NUM_CHANNELS = 1 
NUM_LABELS = 10
# 第一个卷积层中过滤器的尺寸和深度 
CONV1_DEEP =32 
CONV1_SIZE = 5 
# 第二个卷积层中过滤器的尺寸和深度 
CONV2_DEEP = 64 
CONV2_SIZE = 5 
# 全连接层的节点个数 
FC_SIZE = 512

slim = tf.contrib.slim   
# 加载 slim 库 … 
def lenet5(input_tensor): 
    inputs = tf.reshape(input_tensor, [-1, 28, 28, 1]) 
    # 定义一个卷积层，深度为 32 ，过滤器大小为 5 x5 ，使用全 0 填充 
    net = slim.conv2d(inputs, 32, [5, 5], padding='SAME', scope='layer1-conv') 
    # 定义一个最大池化层，其过滤器大小为 2 x2 ，步长为 2 
    net = slim.max_pool2d(net, 2, stride=2, scope='layer2-max-pool') 
    # 类似地定义其他网络层结构 
    net = slim.conv2d(net, 64, [5, 5], padding='SAME', scope='layer3-conv') 
    net = slim.max_pool2d(net, 2, stride=2, scope='layer4-max-pool') 
    # 直接使用 TensorFlow 封装好的 flatten 函数将 4 维矩阵变为 2 维，方便全连接层的计算 
    net = slim.flatten(net, scope='flatten') 
    # 定义全连接层，该全连接层有 500 个隐藏节点 
    net = slim.fully_connected(net, 500, scope='layer5') 
    net = slim.fully_connected(net, 10, scope='output') 
    return net


def inference_slim(input_tensor,train,regularizer): 
    with tf.variable_scope('layer1-conv1'): 
        conv1=slim.conv2d(input_tensor, CONV1_DEEP, [CONV1_SIZE, CONV1_SIZE], padding='SAME', scope='layer1-conv')

    with tf.name_scope('layer2-pool1'): 
        pool1 = slim.max_pool2d(conv1, [2,2],scope='layer2-max-pool') 
    
    with tf.variable_scope('inception-v3'): 
        # 为 Inception 模块中每一条路径声明一个命名空间 
        with tf.variable_scope('Branch_0'): 
            # 实现一个过滤器边长为 1 ，深度为 320 的卷积层 
            branch_0 = slim.conv2d(pool1, 320, [1, 1], scope='Conv2d_2a_1x1') 
        with tf.variable_scope('Branch_1'): 
            branch_1 = slim.conv2d(pool1, 384, [1, 1], scope='Conv2d_2a_1x1') 
        #tf.concat 函数可以将多个矩阵拼接起来， 3 表示在深度维度上进行拼接 
            branch_1 = tf.concat([ # 如图所示，此处 2 层卷积层的输入都是 branch_1 而不是 net 
            slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_2b_1x3'), 
            slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_2c_3x1') ], 3)
        with tf.variable_scope('Branch_2'): 
            branch_2 = slim.conv2d(pool1, 448, [1, 1], scope='Conv2d_2a_1x1') 
            branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_2b_3x3') 
            branch_2 = tf.concat([ 
                slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_2c_1x3'), 
                slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_2d_3x1') ], 3) 
            with tf.variable_scope('Branch_3'): 
                branch_3 = slim.avg_pool2d(pool1, [3, 3], stride=1,padding='SAME', scope='AvgPool_2a_3x3') 
                branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_2d_1x1') 
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)


    with tf.variable_scope('layer3-conv2'): 
        conv2=slim.conv2d(net, CONV2_DEEP, [CONV2_SIZE, CONV2_SIZE], padding='SAME', scope='layer3-conv')
    
    with tf.name_scope('layer4-pool2'): 
        pool2 = slim.max_pool2d(conv2, [2,2],scope='layer4-max-pool')

    # 将第四层池化层的输出转化为第五层全连接层的输入格式 
    #pool2.get_shape 函数可以得到第四层输出矩阵的维度而不需要手工计算 
    pool_shape = pool2.get_shape().as_list() 
    # 将这个 7x7x16 的矩阵拉直成一个向量 
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3] 
    #pool_shape[0] 代表一个 batch 中数据的个数 
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    with tf.variable_scope('layer5-fc1'): 
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.1)) 
        if regularizer != None: 
            tf.add_to_collection('losses', regularizer(fc1_weights)) 
        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1)) 
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases) 
        if train: 
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer6-fc2'): 
        fc2_weights = tf.get_variable("weight", [FC_SIZE, NUM_LABELS],  initializer=tf.truncated_normal_initializer(stddev=0.1)) 
        if regularizer != None: 
            tf.add_to_collection('losses', regularizer(fc2_weights)) 
        fc2_biases = tf.get_variable("bias", [NUM_LABELS], initializer=tf.constant_initializer(0.1)) 
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases 
    return logit




def inference(input_tensor,train,regularizer): 
    with tf.variable_scope('layer1-conv1'): 
        conv1_weights = tf.get_variable("weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP], initializer=tf.truncated_normal_initializer(stddev=0.1)) 
        conv1_biases = tf.get_variable("biases", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME') 
        relu1= tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope('layer2-pool1'): 
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    with tf.variable_scope('layer3-conv2'): 
        conv2_weights = tf.get_variable('weight',  [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP], initializer=tf.truncated_normal_initializer(stddev=0.1)) 
        conv2_biases = tf.get_variable('bias', [CONV2_DEEP],  initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights,  strides=[1, 1, 1, 1], padding='SAME') 
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    
    with tf.name_scope('layer4-pool2'): 
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 将第四层池化层的输出转化为第五层全连接层的输入格式 
    #pool2.get_shape 函数可以得到第四层输出矩阵的维度而不需要手工计算 
    pool_shape = pool2.get_shape().as_list() 
    # 将这个 7x7x16 的矩阵拉直成一个向量 
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3] 
    #pool_shape[0] 代表一个 batch 中数据的个数 
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    with tf.variable_scope('layer5-fc1'): 
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.1)) 
        if regularizer != None: 
            tf.add_to_collection('losses', regularizer(fc1_weights)) 
        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1)) 
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases) 
        if train: 
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer6-fc2'): 
        fc2_weights = tf.get_variable("weight", [FC_SIZE, NUM_LABELS],  initializer=tf.truncated_normal_initializer(stddev=0.1)) 
        if regularizer != None: 
            tf.add_to_collection('losses', regularizer(fc2_weights)) 
        fc2_biases = tf.get_variable("bias", [NUM_LABELS], initializer=tf.constant_initializer(0.1)) 
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases 
    return logit









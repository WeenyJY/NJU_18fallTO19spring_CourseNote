import tensorflow as tf

input_node=784     #输入层节点数
output_node=10      #输出层节点数
layer1_node=250     #隐藏层节点数
layer2_node=250
regularization_rate=0.0001

def inference(input_tensor): 
    with tf.variable_scope('layer1'): 
        weights = tf.get_variable("weights", [input_node, layer1_node], initializer=tf.truncated_normal_initializer(stddev=0.1)) 
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularization_rate)(weights)) 
        biases = tf.get_variable("biases", [layer1_node], initializer=tf.constant_initializer(0.0)) 
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
    with tf.variable_scope('layer2'): 
        weights = tf.get_variable("weights", [layer1_node, layer2_node], initializer=tf.truncated_normal_initializer(stddev=0.1)) 
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularization_rate)(weights)) 
        biases = tf.get_variable("biases", [layer2_node], initializer=tf.constant_initializer(0.0)) 
        layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)
    with tf.variable_scope('layer3'): 
        weights = tf.get_variable("weights", [layer2_node, output_node], initializer=tf.truncated_normal_initializer(stddev=0.1)) 
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularization_rate)(weights)) 
        biases = tf.get_variable("biases", [output_node], initializer=tf.constant_initializer(0.0)) 
        layer3 = tf.matmul(layer2, weights) + biases 

    return layer3
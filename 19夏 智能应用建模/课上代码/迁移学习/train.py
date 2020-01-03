import numpy as np 
import tensorflow as tf 
import tensorflow.contrib.slim as slim
# 加 载 通 过 TensorFlow-Slim 定 义 好 的 inception_v3 模 型 
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3


# 处 理 好 之 后 的 数 据 文 件 
INPUT_DATA = r'C:\Users\Jaqen\Desktop\path\flower_processed_data.npy'
# 保 存 训 练 好 的 模 型 的 路 径 
TRAIN_FILE = r'C:\Users\Jaqen\Desktop\path\save_model'
# 谷 歌 提 供 的 训 练 好 的 模 型 文 件 的 地 址 
CKPT_FILE = r'C:\Users\Jaqen\Desktop\path\save_model\inception_v3.ckpt'

# 定 义 训 练 中 使 用 的 参 数 
LEARNING_RATE = 0.0001 
STEPS = 4
BATCH = 32 
N_CLASSES = 5

# 不 需 要 从 谷 歌 训 练 好 的 模 型 中 加 载 的 参 数 ， 这 里 给 出 的 是 参 数 的 前 缀 
CHECKPOINT_EXCLUDE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits' 
# 需 要 训 练 的 网 络 层 参 数 
TRAINABLE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits'


# 获 取 所 有 需 要 从 谷 歌 训 练 好 的 模 型 中 加 载 的 参 数 
def get_tuned_variables(): 
    exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(',')] 
    variables_to_restore = [] 
    # 枚 举 inception-v3 模 型 中 所 有 的 参 数 ， 然 后 判 断 是 否 需 要 从 加 载 列 表 中 移 除 
    for var in slim.get_model_variables(): 
        excluded = False 
        for exclusion in exclusions: 
            if var.op.name.startswith(exclusion): 
                excluded = True 
                break 
        if not excluded: 
            variables_to_restore.append(var) 
    return variables_to_restore

# 获 取 所 有 需 要 训 练 的 变 量 列 表 
def get_trainable_variables(): 
    scopes = [scope.strip() for scope in TRAINABLE_SCOPES.split(',')] 
    variables_to_train = [] 
    # 枚 举 所 有 需 要 训 练 的 参 数 前 缀 ， 并 通 过 这 些 前 缀 找 到 所 有 的 参 数 
    for scope in scopes: 
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope) 
        variables_to_train.extend(variables) 
    return variables_to_train

def main(): 
    # 加 载 预 处 理 好 的 数 据 
    processed_data= np.load(INPUT_DATA, allow_pickle=True) 
    training_images= processed_data[0] 
    n_training_example= len(training_images) 
    training_labels= processed_data[1] 
    validation_images= processed_data[2] 
    validation_labels= processed_data[3] 
    testing_images= processed_data[4] 
    testing_labels= processed_data[5] 
    print("%d training examples, %d validation examples and %d testing examples." % (n_training_example, len(validation_labels), len(testing_labels)))
    # 定 义 inception-v3 的 输 入 ， images 为 输 入 图 片 ， labels 为 每 一 张 图 片 对 应 的 标 签 
    images = tf.placeholder(tf.float32, [None, 299, 299, 3], name='Input_images')
    labels = tf.placeholder(tf.int64, [None], name='labels')
    
    # 定 义 inception-v3 模 型 
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()): 
        logits, _ = inception_v3.inception_v3(images, num_classes=N_CLASSES)
    # 获 取 需 要 训 练 的 变 量 
    trainable_variables= get_trainable_variables() 
    # 定 义 交 叉 熵 损 失 
    tf.losses.softmax_cross_entropy(tf.one_hot(labels, N_CLASSES), logits, weights=1.0) 
    # 定 义 训 练 过 程 
    train_step= tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(tf.losses.get_total_loss())
    # 计 算 正 确 率 
    with tf.name_scope('evaluation'): 
        correct_prediction= tf.equal(tf.argmax(logits, 1), labels) 
        evaluation_step= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 定 义 加 载 模 型 的 函 数 
    load_fn= slim.assign_from_checkpoint_fn(CKPT_FILE, get_tuned_variables(), ignore_missing_vars=True)

    #定义保存新的训练好的模型的函数 
    saver = tf.train.Saver() 
    with tf.Session() as sess: 
        # 初 始 化 没 有 加 载 进 来 的 变 量 
        init= tf.global_variables_initializer() 
        sess.run(init)
        #加 载 谷 歌 已 经 训 练 好 的 模 型 
        print('Loading tuned variables from%s' % CKPT_FILE) 
        load_fn(sess) 
        start= 0 
        end= BATCH 
        for i in range(STEPS): 
            # 运 行 训 练 过 程 ， 这 里 不 会 更 新 全 部 的 参 数 ， 只 会 更 新 指 定 的 部 分 参 数 
            sess.run(train_step, feed_dict={ images: training_images[start:end], labels: training_labels[start:end] })
            # 输 出 日 志 
            if i% 2 == 0 or i+ 1 == STEPS: 
                #saver.save(sess, TRAIN_FILE, global_step = i)
                validation_accuracy= sess.run(evaluation_step, feed_dict={ images: validation_images, labels: validation_labels }) 
                print('Step %d: Validation accuracy = %.lf%%' % (i, validation_accuracy* 100.0))
            # 因 为 在 数 据 预 处 理 的 时 候 已 经 做 过 了 打 乱 数 据 的 操 作 ， 所 以 这 里 只 需 要 顺 序 使 用 训 练 数 据 
            start = end 
            if start == n_training_example: 
                start = 0 
            end = start + BATCH 
            if end > n_training_example: 
                end = n_training_example
        # 在 后 的 测 试 数 据 上 测 试 正 确 率 
        test_accuracy= sess.run(evaluation_step, feed_dict={ images: testing_images, labels: testing_labels }) 
        print('Final test accuracy = %.lf%%' % (test_accuracy* 100))

if __name__ =='__main__': 
    main()








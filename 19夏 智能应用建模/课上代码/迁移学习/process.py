import glob 
import os.path 
import numpy as np 
import tensorflow as tf 
from tensorflow import gfile


# 原 始 输 入 数 据 的 目 录 
INPUT_DATA = r'C:\Users\Jaqen\Desktop\flower_photos' 
# 输 出 文 件 地 址 ， 将 整 理 后 的 图 片 数 据 通 过 numpy 的 格 式 保 存 
OUTPUT_FILE = r'C:\Users\Jaqen\Desktop\path\flower_processed_data.npy'
# 测 试 数 据 和 验 证 数 据 的 比 例 
VALIDATION_PERCENTAGE = 10 
TEST_PERCENTAGE = 10

# 读 取 数 据 并 将 数 据 分 割 成 训 练 数 据 、 验 证 数 据 和 测 试 数 据 
def create_image_lists(sess, testing_percentage, validation_percentage): 
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)] 
    is_root_dir = True
    # 初 始 化 各 个 数 据 集 
    training_images = [] 
    training_labels = [] 
    testing_images = [] 
    testing_labels = [] 
    validation_images = [] 
    validation_labels = [] 
    current_label = 0

    # 读 取 所 有 的 子 目 录 
    for sub_dir in sub_dirs: 
        if is_root_dir: 
            is_root_dir= False 
            continue
        # 获 取 一 个 子 目 录 中 所 有 的 图 片 文 件 
        extensions= ['jpg', 'jpeg', 'JPG', 'JPEG'] 
        file_list= [] 
        dir_name= os.path.basename(sub_dir) 
        for extension in extensions: 
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension) 
            #glob.glob ： 返 回 所 有 匹 配 的 文 件 路 径 列 表 
            file_list.extend(glob.glob(file_glob)) 
        if not file_list: 
            continue 
        size = 0
        # 处 理 图 片 数 据 
        for file_name in file_list: 
            # 读 取 并 解 析 图 片 ， 将 图 片 转 化 为 299x299 以 便 inception-v3 模 型 来 处 理 
            image_raw_data= gfile.FastGFile(file_name, 'rb').read() 
            image = tf.image.decode_jpeg(image_raw_data) 
            if image.dtype!= tf.float32: 
                image = tf.image.convert_image_dtype(image, dtype=tf.float32) 
            image = tf.image.resize_images(image, [299, 299]) 
            image_value= sess.run(image)
            # 随 机 划 分 数 据 集 
            chance = np.random.randint(100) 
            if chance < validation_percentage: 
                validation_images.append(image_value) 
                validation_labels.append(current_label) 
            elif chance < (testing_percentage+ validation_percentage): 
                testing_images.append(image_value) 
                testing_labels.append(current_label)
            else:  
                training_images.append(image_value) 
                training_labels.append(current_label) 
            size+= 1 
            # 每 种 花 只 取 100 张 图 片 ， 这 样 可 以 缩 短 程 序 的 运 行 时 间 
            if size== 100: 
                break 
        current_label+= 1 
        print("current_label: %d" % current_label) 

    # 将 训 练 数 据 随 机 打 乱 以 获 得 更 好 的 训 练 效 果 
    state= np.random.get_state() 
    np.random.shuffle(training_images) 
    np.random.set_state(state) 
    np.random.shuffle(training_labels) 
    return np.asarray([training_images, training_labels,validation_images, validation_labels,testing_images, testing_labels])

def main(): 
    with tf.Session() as sess: 
        processed_data= create_image_lists(sess, TEST_PERCENTAGE, VALIDATION_PERCENTAGE) 
        np.save(OUTPUT_FILE, processed_data) 

if __name__ == '__main__': 
    main()
























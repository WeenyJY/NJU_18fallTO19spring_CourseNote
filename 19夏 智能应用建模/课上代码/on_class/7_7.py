#encoding: utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 

image_raw_data = tf.gfile.FastGFile(r'C:\Users\Jaqen\Desktop\智能应用建模\picture\cat.jpg', 'rb').read()
# 给 定 一 张 图 像 ， 随 机 调 整 图 像 的 色 彩 。 因 为 调 整 亮 度 、 对 比 度 、 饱 和 度 和 色 相 
# 的 顺 序 会 影 响 后 得 到 的 结 果 ， 所 以 可 以 定 义 多 种 不 同 的 顺 序 。 具 体 使 用 哪 一 
# 种 顺 序 可 以 在 训 练 数 据 预 处 理 时 随 机 地 选 择 一 种 ， 这 样 可 以 进 一 步 降 低 无 关 因素对模型的影响
def distort_color(image, color_ordering=0): 
    if color_ordering == 0: 
        image = tf.image.random_brightness(image, max_delta=32. / 255.)   #亮度
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)    #饱和度
        image = tf.image.random_hue(image, max_delta=0.2)                #色相
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)    #对比度
    elif color_ordering == 1: 
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5) 
        image = tf.image.random_brightness(image, max_delta=32. / 255.) 
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5) 
        image = tf.image.random_hue(image, max_delta=0.2) 
    return tf.clip_by_value(image, 0.0, 1.0)

def preprocess_for_train(image, height, width, bbox): 
    # 如 果 没 有 提 供 标 注 框 ， 则 认 为 整 个 图 像 就 是 需 要 关 注 的 部 分 
    if bbox is None: 
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4]) 
    # 转 换 图 像 张 量 的 类 型 
    if image.dtype != tf.float32: 
        image = tf.image.convert_image_dtype(image, dtype=tf.float32) 
    # 随 机 截 取 图 像 ， 减 少 需 要 关 注 的 物 体 大 小 对 图 像 识 别 算 法 的 影 响 
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=bbox, min_object_covered=0.4) 
    distorted_image = tf.slice(image, bbox_begin, bbox_size) 
    # 将 随 机 裁 取 的 图 像 调 整 为 神 经 网 络 输 入 层 的 大 小 。 随 机 选 择 大 小 调 整 算 法 
    distorted_image = tf.image.resize_images( distorted_image, [height, width], method=np.random.randint(4)) 
    # 随 机 左 右 翻 转 图 像 
    distorted_image = tf.image.random_flip_left_right(distorted_image) 
    # 使 用 一 种 随 机 的 顺 序 调 整 图 像 色 彩 
    distorted_image = distort_color(distorted_image, np.random.randint(2)) 
    return distorted_image


with tf.Session() as sess: 
    img_data = tf.image.decode_jpeg(image_raw_data)    
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
    # 运 行 6 次 获 得 6 种 不 同 的 图 像 
    for i in range(6): 
        # 将 图 像 的 尺 寸 调 整 为 299x299 
        result = preprocess_for_train(img_data, 299, 299, boxes) 
        plt.imshow(result.eval()) 
        plt.show()



















    '''
    # 将 图 片 数 据 转 化 为 实 数 类 型 
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
    # 调 整 图 像 大 小 ， method 取 值 # 0 ： 双 线 性 插 值 法 ， 1 ： 近 邻 居 法 ， 2 ： 双 三 次 插 值 法 ， 3 ： 面 积 插 值 法 
    img_data = tf.image.resize_images(img_data, [300, 300], method=0)
    plt.imshow(img_data.eval()) 
    plt.show() 
    '''
    '''
    # 将 图 像 缩 小 一 些 ， 这 样 可 视 化 能 让 标 注 框 更 加 清 楚 
    img_data = tf.image.resize_images(img_data, [180, 267], method=1)
    #tf.image.draw_bounding_boxes 函 数 要 求 图 像 矩 阵 中 的 数 字 为 实 数 ， 需 要 先 将 # 图 像 矩 阵 转 化 为 实 数 类 型 ， 
    #tf.image.draw_bounding_boxes 函 数 的 图 像 输 入 是 一 个 batch 的 数 据 （ 多 张 图 像 组 成 的 4 维 矩 阵 ） ， 
    # 所 有 需 要 将 解 码 之 后 的 矩 阵 加 一 维 
    batched = tf.expand_dims(tf.image.convert_image_dtype(img_data, tf.float32), 0)
    # 给 出 每 一 张 图 像 的 所 有 标 注 框 # 一 个 标 注 框 有 4 个 数 字 ， 分 别 代 表 [y_min, x_min, y_max, x_max] 
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
    # 在 图 像 中 加 入 标 注 框 
    result = tf.image.draw_bounding_boxes(batched, boxes)
    plt.imshow(result[0].eval()) 
    plt.show()
    '''
    '''
    # 通 过 标 注 框 可 视 化 随 机 截 取 得 到 的 图 像 
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]]) 
    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box( tf.shape(img_data), 
                        bounding_boxes=boxes, min_object_covered=0.4 )# 表 示 截 取 部 分 至 少 包 含 某 个 标 注 框 40% 的 内 容 
    # 通 过 标 注 框 可 视 化 随 机 截 取 得 到 的 图 像 
    batched = tf.expand_dims(tf.image.convert_image_dtype(img_data, tf.float32), 0) 
    image_with_box = tf.image.draw_bounding_boxes(batched, bbox_for_draw)
    # 随 机 截 取 出 来 的 图 像 。 因 有 随 机 成 分 ， 每 次 得 到 的 结 果 会 有 所 不 同 
    distorted_image = tf.slice(img_data, begin, size)
    plt.imshow(distorted_image.eval()) 
    plt.show()
    '''







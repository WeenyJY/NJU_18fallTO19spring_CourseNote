#-*-coding:utf8-*-

import os
import cv2
import numpy as np
import json
import pickle
from read_img import endwith


#输入一个文件路径，对其下的每个文件夹下的图片读取，并对每个文件夹给一个不同的Label
#返回一个img的list,返回一个对应label的list,返回一下有几个文件夹（有几种label)

def read_file(path):
    img_list = []
    label_list = []
    dir_counter = 0
    IMG_SIZE = 128
    file_count = 0
    labbel={}
    #对路径下的所有子文件夹中的所有jpg文件进行读取并存入到一个list中
    for child_dir in os.listdir(path):
        if child_dir == '.DS_Store':
            continue
        child_path = os.path.join(path, child_dir)

        for child_dir2 in os.listdir(child_path):
             if child_dir2 == '.DS_Store':
                 continue
             child_path2 = os.path.join(child_path,child_dir2)
             num=20
             for dir_image in os.listdir(child_path2):
                 if dir_image=='.DS_Store':
                     continue
                 if dir_image[-3:]=='gif':
                     continue
                 num-=1
                 #if num>0:
                 #    print(child_path2)
                 file_count += 1
                 #print(dir_image)
                 if endwith(dir_image,'jpg'):
                    img = cv2.imread(os.path.join(child_path2, dir_image))
                    #print(img)
                    resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    recolored_img = cv2.cvtColor(resized_img,cv2.COLOR_BGR2GRAY)
                    img_list.append(recolored_img)
                    label_list.append(dir_counter)
                    labbel[dir_counter]=dir_image[:-6]
             dir_counter += 1
    with open('label.txt', 'wb') as file:
        #file.write(json.dumps(labbel))
        pickle.dump(labbel, file)
        # 返回的img_list转成了 np.array的格式
    img_list = np.array(img_list)
    return img_list, label_list, dir_counter
    #return img_list,label_list,dir_counter,file_count


#useless
#读取训练数据集的文件夹，把他们的名字返回给一个list
def read_name_list(path):
    name_list = []
    for child_dir in os.listdir(path):
        name_list.append(child_dir)
    return name_list



if __name__ == '__main__':
    img_list,label_list,dir_counter = read_file('../faces94')
    print(dir_counter)
    print(img_list.shape)



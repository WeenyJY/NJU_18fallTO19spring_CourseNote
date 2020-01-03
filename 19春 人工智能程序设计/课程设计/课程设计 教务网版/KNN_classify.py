import numpy as np
import pandas as pd
from PIL import Image
import os

import pic_cut as cut
import norm_pca as norm

class KNN(object):
    def __init__(self):
        pass
    def fit(self,X_list,Y,n_neighbors):
        self.X_list=X_list
        self.Y=Y
        self.n_neighbors=n_neighbors
    def distance(self,x,X):      #用L1范数来定义矩阵间的距离,x为待测图片矩阵，X为数据库中矩阵
        dis=x-X
        L1_norm=0
        for i in range(dis.shape[0]):
            for j in range(dis.shape[1]):
                L1_norm+=abs(dis[i][j])
        return L1_norm
    def predict(self,x):
        dis=[]    #dis为到每一个样本的距离
        count={}  #储存前k个最小距离中不同label的个数
        for i in range(self.X_list.shape[0]):
            X=self.X_list[i]                         
            dis.append(self.distance(x,X))          
        sort_index=np.array(dis).argsort()          #储存从小到大排序后 原来的索引          
        for j in range(self.n_neighbors):
            label=self.Y[sort_index[j]]
            count[label]=count.get(label,0)+1        
        max_count=0
        for key,value in count.items():
            if value>max_count:
                max_count=value
                max_num_label=key 
        predict=max_num_label
        return predict
    def if_true(self,x,label):   #是否正确，label为真实属性
        y=self.predict(x)
        return y==label

'''x_list=np.load('C:/Users/Jaqen/Desktop/matrix_list1.npy')  
Y=np.load('C:/Users/Jaqen/Desktop/class_list1.npy')


path = r'C:/Users/Jaqen/Desktop'
from_dir = 'justfortest'

knn=KNN()
knn.fit(x_list,Y,40)


for photo in os.listdir(path+'/'+from_dir):  
    img = Image.open(path+'/'+from_dir+'/'+photo)
    letters=cut.cut(img)
    n=0
    for i,j in letters:
        im=img.crop((i,0,j,img.size[1]))
        X=norm.normalize(im)
        pca=norm.PCA(X,n_components=3)
        X_=pca.reduce_dim()   #降维
        print(knn.predict(X_),photo[n],knn.if_true(X_,photo[n]))
        n+=1'''



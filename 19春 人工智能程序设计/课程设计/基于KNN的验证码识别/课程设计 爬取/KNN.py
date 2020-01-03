import numpy as np
import pandas as pd

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


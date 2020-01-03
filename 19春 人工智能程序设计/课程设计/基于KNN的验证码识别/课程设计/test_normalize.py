from PIL import Image
import numpy as np
import pandas as pd


def get_pixel_matrix(img):       #获取像素矩阵
    array=np.zeros((img.size[1],img.size[0]))
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            if img.getpixel((x,y))==255:
                array[y,x]=1
            else:
                array[y,x]=0
    return array

def normalize(img):           #规范化为7*23矩阵
    x=img.size[0]
    mat=get_pixel_matrix(img)
    if x<7:
        b=np.ones((1,img.size[1]))
        for i in range(7-x):
            mat=np.insert(mat,0,values=b,axis=1)
    elif x>7:
        for i in range(x-7):
            mat=np.delete(mat,0 if i%2==0 else mat.shape[1]-1,axis=1)
    return mat

class PCA(object):
#定义PCA类，传入的n*m矩阵为n条数据m维
    def __init__(self,X,n_components):
        mean=X.mean(axis=0)
        X=X-mean #中心化后的矩阵
        self.X=X.T
        self.n_components=n_components
        self.datapoints=self.X.shape[1]
    def cov(self):
        return np.dot(self.X,self.X.T)/self.datapoints  #返回协方差矩阵
    def eigenvalues_vectors(self):
        x_cov=self.cov()
        a,b=np.linalg.eig(x_cov)  #特征值与特征向量
        c=np.hstack([a.reshape(-1,1),b.T])  
        c_df=pd.DataFrame(c).sort_values(by=0,ascending=False)
        return c_df   
    def reduce_dim(self): #n_components为降成的维数 
        x_df=self.eigenvalues_vectors().values[:self.n_components,1:]  #x_df为以特征值为行向量的矩阵
        return np.dot(x_df,self.X).T
    def explained_variance(self): #主成分方差值
        return np.var(self.reduce_dim(),axis=0)
    def explained_variance_ratio(self):  #主成分方差比例
        return self.explained_variance()/self.explained_variance().sum()
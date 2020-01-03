from PIL import Image
import numpy as np
import pandas as pd
import os

def get_pixel_matrix(img):       #获取像素矩阵
    array=np.zeros((img.size[1],img.size[0]))
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            if img.getpixel((x,y))==255:
                array[y,x]=1
            else:
                array[y,x]=0
    return array

def normalize(img):           #规范化为15*20矩阵
    x=img.size[0]
    mat=get_pixel_matrix(img)
    if x<15:
        b=np.ones((1,img.size[1]))
        for i in range(15-x):
            mat=np.insert(mat,0,values=b,axis=1)
    elif x>15:
        i=x-15
        if i==1:
            mat=np.delete(mat,0,axis=1)
        elif i==2:
            mat=np.delete(mat,0,axis=1)
            mat=np.delete(mat,mat.shape[1]-1,axis=1)
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


'''path = r'C:/Users/Jaqen/Desktop'
from_dir = 'cut'

matrix_list=[]      #存储所有标准样本的矩阵
class_list=[]       #矩阵所对应的字母/数字
for photo in os.listdir(path+'/'+from_dir): 
    img = Image.open(path+'/'+from_dir+'/'+photo)
    X=normalize(img)
    pca=PCA(X,n_components=3)
    X_=pca.reduce_dim()   #降维
    matrix_list.append(X_)
    class_list.append(photo[0])

np.save('C:/Users/Jaqen/Desktop/matrix_list1.npy',matrix_list)
np.save('C:/Users/Jaqen/Desktop/class_list1.npy',class_list)

#u=np.load('C:/Users/Jaqen/Desktop/matrix_list.npy')    读取文件操作
#z=np.load('C:/Users/Jaqen/Desktop/class_list.npy')'''


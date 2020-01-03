import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def MLR(X,Y,alpha,interations):#学习率与迭代次数
    X=np.hstack([X,np.ones((X.shape[0],1))])
    W=np.zeros((X.shape[1],1))
    C=Y-X.dot(W)
    m=Y.shape[0]
    J=np.dot(C.T,C)/(2*m) #J的导数为dr_J=X_norm.T.dot(X.dot(W)-Y)/m
    for i in range(interations):
        dr_J=X.T.dot(X.dot(W)-Y)/m
        W=W-alpha*dr_J
    return W

wine=pd.read_csv('C:/Users/Jaqen/Desktop/python/winequality-red.csv')
x=wine.values[:,-3:] #红酒的三维数据
X=x[:,:2]
Y=x[:,2].reshape(-1,1)
W=MLR(X,Y,alpha=0.01,interations=100000)
print(W)
y=W[0]*X[:,0]+W[1]*X[:,1]+W[2]
f1=plt.figure()
ax=f1.add_subplot(111, projection='3d')
ax.scatter(X[:,0],x[:,1],Y,marker='o',c='b')
ax.plot(X[:,0],X[:,1],y,marker='d',c='r')
plt.show()



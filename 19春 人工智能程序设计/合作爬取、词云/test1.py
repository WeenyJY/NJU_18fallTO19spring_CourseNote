import pandas as pd
import numpy as np
from sklearn import preprocessing

def MLR(X,Y,alpha,interations):
    X=np.hstack([X,np.ones((X.shape[0],1))])
    W=np.zeros((X.shape[1],1))
    X_norm=preprocessing.scale(X,axis=1)
    C=Y-X_norm.dot(W)
    m=Y.shape[0]
    J=np.dot(C.T,C)/(2*m)
    #dr_J=X_norm.T.dot(X.dot(W)-Y)/m
    for i in range(interations):
        dr_J=X_norm.T.dot(X.dot(W)-Y)/m
        W=W-alpha*dr_J
    return W

wine=pd.read_csv('C:/Users/Jaqen/Desktop/python/winequality-red.csv')
x=wine.values[:,-2:] #红酒的三维数据
X=x[:,0].reshape(-1,1)
Y=x[:,1].reshape(-1,1)
W=MLR(X,Y,alpha=0.01,interations=10000)
y=W[0]*X+W[1]

import matplotlib.pyplot as plt
f1=plt.figure()
ax=f1.add_subplot(111)
ax.scatter(X,Y,marker='o',c='b')
ax.plot(X,y,marker='d',c='r')
plt.show()
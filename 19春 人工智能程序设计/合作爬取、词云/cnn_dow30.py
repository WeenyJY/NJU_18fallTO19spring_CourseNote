'''
import requests
import re

def retrieve_dji_list():
    try:
        r = requests.get('http://money.cnn.com/data/dow30/')
    except Exception as err:
        print(err)
    search_pattern = re.compile('class="wsod_symbol">(.*?)<\/a>.*?<span.*?">(.*?)<\/span>.*?\n.*?class="wsod_stream">(.*?)<\/span>')
    dji_list_in_text = re.findall(search_pattern, r.text)
    return dji_list_in_text

dji_list = retrieve_dji_list()
print(dji_list)'''
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

wine=pd.read_csv('C:/Users/Jaqen/Desktop/python/winequality-red.csv')
x=wine.values[:,-3:] #红酒的三维数据
pca=PCA(n_components=2)

y=pca.fit_transform(x) #降维
z=pca.inverse_transform(y)
f1=plt.figure()
f2=plt.figure()
ax=f1.add_subplot(111, projection='3d')
ax.scatter(x[:,0],x[:,1],x[:,2],marker='o',c='b')
ax.scatter(z[:,0],z[:,1],z[:,2],marker='*',c='r')
f1.suptitle('dim-3',va='top')

bx=f2.add_subplot(111,projection=None)
bx.scatter(y[:,0],y[:,1],marker='*',c='r')
f2.suptitle('dim_redu_to_2')

plt.show()



from PIL import Image
import os
import numpy as np
import test_cut as cut
import test_normalize as norm
import test_denoise as denoise
import KNN
import re
import requests

def identify(file):
    
    x_list=np.load('C:/Users/Jaqen/Desktop/matrix_list.npy')  
    Y=np.load('C:/Users/Jaqen/Desktop/class_list.npy')
    knn=KNN.KNN()
    knn.fit(x_list,Y,10)
    img = Image.open(denoise.process(file))
    letters=cut.cut(img)
    pre_letters=''
    for i,j in letters:
        im=img.crop((i,0,j,img.size[1]))
        X=norm.normalize(im)
        pca=norm.PCA(X,n_components=3)  #降维
        X_=pca.reduce_dim()   
        predic=knn.predict(X_)
        pre_letters+=predic
    return pre_letters


URL = 'https://www.quanjing.com/createImg.aspx'
r = requests.get(URL).content
with open('C:/Users/Jaqen/Desktop/test.jpg', 'wb') as f:
    f.write(r)
print(identify('test.jpg'))
img=Image.open('C:/Users/Jaqen/Desktop/test.jpg')
img.show()


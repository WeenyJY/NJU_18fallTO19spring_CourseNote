from PIL import Image
import os
import pic_cut as cut
import norm_pca as norm
import KNN_classify as KNN
import numpy as np
import graying_and_denoise as denoise

x_list=np.load('C:/Users/Jaqen/Desktop/matrix_list1.npy')  
Y=np.load('C:/Users/Jaqen/Desktop/class_list1.npy')

knn=KNN.KNN()
knn.fit(x_list,Y,20)

denoise.process('x.jpg')
img=Image.open(r'C:/Users/Jaqen/Desktop/pic.png')
letters=cut.cut(img)
n=0  
pre_letters=''

for i,j in letters:
    im=img.crop((i,0,j,img.size[1]))
    X=norm.normalize(im)
    pca=norm.PCA(X,n_components=3)
    X_=pca.reduce_dim()   #降维
    predic=knn.predict(X_)
    pre_letters+=predic
print(pre_letters)
'''for photo in os.listdir(path+'/'+from_dir):  
    img = Image.open(path+'/'+from_dir+'/'+photo)
    letters=cut.cut(img)
    n=0  
    pre_letters=''
    single_score=0
    for i,j in letters:
        im=img.crop((i,0,j,img.size[1]))
        X=norm.normalize(im)
        pca=norm.PCA(X,n_components=3)
        X_=pca.reduce_dim()   #降维
        predic=knn.predict(X_)
        single_score+=knn.if_true(X_,photo[n])
        n+=1
        pre_letters+=predic
    score_list.append(single_score/4.0)

total_score=score_list.count(1)
print(score_list)
print(total_score/len(score_list))'''
    

from PIL import Image
import os
import numpy as np
import test_cut as cut
import test_normalize as norm
import test_denoise as denoise
import KNN


x_list=np.load('C:/Users/Jaqen/Desktop/matrix_list.npy')  
Y=np.load('C:/Users/Jaqen/Desktop/class_list.npy')

path = r'C:/Users/Jaqen/Desktop'
test_dir = '数字测试集'

from_dir=denoise.process(test_dir)  #降噪后的测试集文件

knn=KNN.KNN()
knn.fit(x_list,Y,10)

score_list=[]

for photo in os.listdir(path+'/'+from_dir):  
    img = Image.open(path+'/'+from_dir+'/'+photo)
    letters=cut.cut(img)
    n=0  
    pre_letters=''
    single_score=0
    for i,j in letters:
        im=img.crop((i,0,j,img.size[1]))
        X=norm.normalize(im)
        pca=norm.PCA(X,n_components=3)  #降维
        X_=pca.reduce_dim()   
        predic=knn.predict(X_)
        single_score+=knn.if_true(X_,photo[n])
        n+=1
        pre_letters+=predic
    score_list.append(single_score/4.0)

total_score=score_list.count(1)
print(score_list)
print('0%:',score_list.count(0)/len(score_list))
print('25%:',score_list.count(0.25)/len(score_list))
print('50%:',score_list.count(0.5)/len(score_list))
print('75%:',score_list.count(0.75)/len(score_list))
print('100%:',total_score/len(score_list))
    
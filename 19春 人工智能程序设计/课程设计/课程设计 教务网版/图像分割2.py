from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

def get_pixel_matrix(img):       #获取像素矩阵
    array=np.zeros(img.size)
    print(array.size)
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            array[x,y]=img.getpixel((x,y))
    return array


def check_filename_available(path,file_name,m):   #判断文件名是否重复并生成新文件名
    file_name_new=file_name
    if os.path.isfile(path+'/'+file_name):
        file_name_new=file_name[0]+'_'+str(m)+file_name[-4:]
        m+=1
    if os.path.isfile(path+'/'+file_name_new):
        file_name_new=check_filename_available(path,file_name_new,m)   #递归判断
    return file_name_new

def vertical(img):      #获取垂直投影
    pixel_data = img.load()
    vertical_shadow = []
    for x in range(img.size[0]):
        black = 0
        for y in range(img.size[1]):
            if pixel_data[x,y] == 0:
                black += 1
        vertical_shadow.append(black)
    return vertical_shadow    #1*img.size[0]

def cut(img):
    vertical_shadow=vertical(img)
    pre_index=[]  #储存预选分割点
    final_index=[]
    for i in range(len(vertical_shadow)):
        if vertical_shadow[i]<=2:    #小于等于2的都认作预选点
            pre_index.append(i)
    
    pre_index_diff=[0]+[pre_index[i]-pre_index[i-1] for i in range(1,len(pre_index))]

    for i in range(len(pre_index_diff)):
        if pre_index_diff[i]>=5:
            final_index.append((pre_index[i-1],pre_index[i]))
        if len(final_index)==4:
            break
    return final_index




    '''while i < range(len(pre_index)):
        for j in range(i+1,len(pre_index)):
            if pre_index[j]-pre_index[i]>=5:   #宽度大于=5可认为是数字/字母
                final_index.append((i,j))'''
                





    




img=Image.open('C:/Users/Jaqen/Desktop/二值化降噪/2BPP.png')
ver=vertical(img)
plt.bar(range(img.size[0]),ver)
#plt.show()
print(cut(img))
letters=cut(img)
for i,j in letters:
    im=img.crop((i,0,j,img.size[1]))
    im.show()









'''path = r'C:/Users/Jaqen/Desktop'
from_dir = 'photo1'
to_dir = '切割'

for photo in os.listdir(path+'/'+from_dir):  
    img = Image.open(path+'/'+from_dir+'/'+photo)
    letters=cut(img)
    n=0
    for i,j in letters:
        im=img.crop((i,0,j,img.size[1]))  #切割，(left, upper, right, lower)
        filename=check_filename_available(path+'/'+to_dir,photo[n]+'_0'+photo[-4:],0)
        im.save(path+'/'+to_dir+'/'+filename)
        n+=1'''
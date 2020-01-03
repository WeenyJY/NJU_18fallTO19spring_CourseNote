from PIL import Image
import numpy as np
import os

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

def cut_2(img):   #切割粘连字母
    ver=vertical(img)
    middle = len(ver) // 2 
    temp = ver[middle-2:middle+3]   #左右取五个点
    max_dark=temp.index(max(temp))+middle-2  #黑点最多的列index
    max_white=temp.index(min(temp))+middle-2 #白点最多的列index
    if abs(max_dark-len(ver)//2)<abs(max_white-len(ver)//2):
        cut_x=max_dark
    else:
        cut_x=max_white
    return cut_x

def check_letters(letter):       #检查切割点列表，若有大于四个切割处则进行一些操作
    cur_letters=letter
    if len(letter)==4:
        temp=[x[1]-x[0] for x in letter]
        for i in range(4):
            if temp[i]<5:
                cur_letters[i]=(letter[i][0],letter[i][0]+6)
            elif temp[i]>16:
                cur_letters[i]=(letter[i][0],letter[i][0]+9)
        return cur_letters,True
    elif len(letter)>4:
        final_letters=[]
        temp=[x[1]-x[0] for x in letter]
        i=0
        while i < len(temp):
            if i<len(temp)-1 and temp[i]<5:
                if temp[i+1]<5:
                    final_letters.append((letter[i][0],letter[i+1][1]))
                    i+=2
                else:
                    final_letters.append(letter[i])
                    i+=1
            elif i==len(letter)-1 and temp[i]<5:
                final_letters.remove(final_letters[-1])
                final_letters.append((letter[i-1][0],letter[i][1]))
                i+=1
            else:
                final_letters.append(letter[i])
                i+=1
        cur_letters=final_letters
        return cur_letters,False
    return cur_letters,True

def cut(img):
    touchletter=False  #刚接触到数字
    inletter=False     
    letters=[]

    for x in range(img.size[0]):    #im.size[0]为列数，im.size[1]为行数
        count1=0
        count2=0
        for y in range(img.size[1]):
            if img.getpixel((x,y))!=255: #不是白的，开始接触数字,count用来记录某一列黑点的个数
                count1+=1
            if x<img.size[0]-1:
                if img.getpixel((x+1,y))!=255:
                    count2+=1
        if count1/img.size[1]>=0.1 and count2/img.size[1]>=0.1:  #阈值，某一列黑点数大于阈值且前一列大于阈值则可认为是数字而不是噪声
            touchletter=True           
        if inletter==False and touchletter==True:
            start=x
            inletter=True
        if inletter==True and touchletter==False :
            end=x+1
            if end-start>=17:          #宽度过长，出现粘连情况
                mid_x=cut_2(img.crop((start,0,end,img.size[0])))+start
                letters.append((start,mid_x+1))
                letters.append((mid_x-1,end))
            else:
                letters.append((start,end))
            inletter=False            
        touchletter=False 
    letters,boolean=check_letters(letters)
    times=1
    while boolean==False and times<5:
        letters,boolean=check_letters(letters)
        times+=1

    return letters
    
def check_filename_available(path,file_name,m):   #判断文件名是否重复并生成新文件名
    file_name_new=file_name
    if os.path.isfile(path+'/'+file_name):
        file_name_new=file_name[0]+'_'+str(m)+file_name[-4:]
        m+=1
    if os.path.isfile(path+'/'+file_name_new):
        file_name_new=check_filename_available(path,file_name_new,m)   #递归判断
    return file_name_new

'''path = r'C:/Users/Jaqen/Desktop'
from_dir = '降噪后'
to_dir = 'cut'

for photo in os.listdir(path+'/'+from_dir):  
    img = Image.open(path+'/'+from_dir+'/'+photo)
    letters=cut(img)
    n=0
    for i,j in letters:
        im=img.crop((i,0,j,img.size[1]))  #切割，(left, upper, right, lower)
        filename=check_filename_available(path+'/'+to_dir,photo[n]+'_0'+photo[-4:],0)
        im.save(path+'/'+to_dir+'/'+filename)
        n+=1'''

'''img = Image.open(r'C:/Users/Jaqen/Desktop/8w38.png')
letters=cut(img)
n=0
for i,j in letters:
    im=img.crop((i,0,j,img.size[1]))  #切割，(left, upper, right, lower)
    filename=check_filename_available(r'C:/Users/Jaqen/Desktop','8w38.png'[n]+'_0'+'8w38.png'[-4:],0)
    im.save(r'C:/Users/Jaqen/Desktop'+'/'+filename)
    n+=1'''

 
import os, sys
from PIL import Image
   
def depoint_1(img):
   pixdata = img.load()
   w,h = img.size
   yu = 100
   for y in range(0,h):
      for x in range(0,w):
         count = 0
         if x == w-1:
            pixdata[x, y] = 255
         if x == 0:
            pixdata[x, y] = 255
         if y == 0:
            pixdata[x, y] = 255
         if y ==h-1:
            pixdata[x, y] = 255 
         if 0<x<w-1 & 0<y<h-1:
            if pixdata[x,y-1] > yu:#上
               count += 1
            if pixdata[x,y+1] > yu:#下
               count += 1
            if pixdata[x-1,y] > yu:#左
               count += 1
            if pixdata[x+1,y] > yu:#右
               count += 1
            if pixdata[x-1,y-1] > yu:#左上
               count += 1
            if pixdata[x-1,y+1] > yu:#左下
               count += 1
            if pixdata[x+1,y-1] > yu:#右上
               count += 1
            if pixdata[x+1,y+1] > yu:#右下
               count += 1
            if count > 6:
               pixdata[x,y] = 255
            if pixdata[x,y] > 100:
               pixdata[x,y] = 255
   return img

def depoint_2(img):
    pixdata = img.load()
    w,h = img.size
    yu = 100
    for y in range(1,h-1):
        for x in range(1,w-1):
            count = 0
            if pixdata[x,y-1] > yu:#上
                count += 1
            if pixdata[x,y+1] > yu:#下
                count += 1
            if pixdata[x-1,y] > yu:#左
                count += 1
            if pixdata[x+1,y] > yu:#右
                count += 1
            if pixdata[x-1,y-1] > yu:#左上
                count += 1
            if pixdata[x-1,y+1] > yu:#左下
                count += 1
            if pixdata[x+1,y-1] > yu:#右上
                count += 1
            if pixdata[x+1,y+1] > yu:#右下
                count += 1
            if count > 7:
                pixdata[x,y] = 255
    return img

def process(file):
   path = r'C:/Users/Jaqen/Desktop'
   from_jpg_dir = file
   to_dir = '降噪后测试集'
   from_dir = '降噪后测试集'

   photos = os.listdir(path + '/' + from_jpg_dir)
   for photo in photos:
      im = Image.open(path + '/' + from_jpg_dir+'/'+photo)
      im.save(path + '/' + to_dir+'/'+photo[0:4]+'.png')

   for i in range(2):
      photos = os.listdir(path + '/' + from_dir)
      for photo in photos:
    
         img = Image.open(path + '/' + from_dir + '/' + photo)
         img1 = depoint_1(img) 
         img1.save(path + '/' + to_dir + '/' + photo)
   
      photos = os.listdir(path + '/' + to_dir)
      for photo in photos:
    
         img = Image.open(path + '/' + to_dir + '/' + photo)
         img1 = img.convert('1')
         if i==1:
            img1 = depoint_2(img1) 
            img1.save(path + '/' + to_dir + '/' + photo)
         else:
            img1.save(path + '/' + to_dir + '/' + photo)


'''path = r'C:/Users/Jaqen/Desktop'
from_jpg_dir = '数字训练集'
to_dir = '降噪数字训练集'
from_dir = '降噪数字训练集'

photos = os.listdir(path + '/' + from_jpg_dir)
for photo in photos:
   im = Image.open(path + '/' + from_jpg_dir+'/'+photo)
   im.save(path + '/' + to_dir+'/'+photo[0:4]+'.png')

for i in range(2):
   photos = os.listdir(path + '/' + from_dir)
   for photo in photos:
    
      img = Image.open(path + '/' + from_dir + '/' + photo)
      img1 = depoint_1(img) 
      img1.save(path + '/' + to_dir + '/' + photo)
   
   photos = os.listdir(path + '/' + to_dir)
   for photo in photos:
    
      img = Image.open(path + '/' + to_dir + '/' + photo)
      img1 = img.convert('1')
      if i==1:
         img1 = depoint_2(img1) 
         img1.save(path + '/' + to_dir + '/' + photo)
      else:
         img1.save(path + '/' + to_dir + '/' + photo)'''

      
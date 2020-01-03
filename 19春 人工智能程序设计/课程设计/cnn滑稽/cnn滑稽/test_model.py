

from read_data import read_name_list,read_file
from train_model import Model
import cv2
import pickle

labbel={}

def test_onePicture(path):
    model= Model()
    model.load()
    img = cv2.imread(path)
    #print(img)
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    picType,prob = model.predict(img)
    if picType != -1:
        #print(picType,prob)
        #name_list = read_name_list('../myAIlab/pictures/dataset')
        #print(name_list[picType],prob)
        print(labbel[picType],prob)
    else:
        print("invaild person")

#读取文件夹下子文件夹中所有图片进行识别
def test_onBatch(path):
    model= Model()
    model.load()
    index = 0
    img_list, label_lsit, counter = read_file(path)
    for img in img_list:
        picType,prob = model.predict(img)
        if picType != -1:
            index += 1
            print(labbel[picType], prob)
        else:
            print("invaild person")

    return index

if __name__ == '__main__':
    with open('label.txt', 'rb') as file:
        #print(file.read())
        #str=file.read()

        labbel =pickle.load(file)
    test_onePicture('guess/anonym.1.jpg')




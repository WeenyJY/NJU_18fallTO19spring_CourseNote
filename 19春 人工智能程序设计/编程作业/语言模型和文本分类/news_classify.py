import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
  
 
def error_rate(y_predict,y):   #错误率
    count=0
    for i in range(y_predict.shape[0]):
        if y_predict[i]!=y[i]:
            count+=1
    return count/y_predict.shape[0]

#读取文件
path1=r'C:/Users/Jaqen/Desktop/News Topic Classification/train_texts.txt'
path2=r'C:/Users/Jaqen/Desktop/News Topic Classification/train_labels.txt'
path3=r'C:/Users/Jaqen/Desktop/News Topic Classification/test_texts.txt'
path4=r'C:/Users/Jaqen/Desktop/News Topic Classification/test_labels.txt'

with open(path1) as f:
    train_texts=f.read().splitlines()
with open(path2) as f:
    train_labels=f.read().splitlines()
with open(path3) as f:
    test_texts=f.read().splitlines()
with open(path4) as f:
    test_labels=f.read().splitlines()


#将文档数据转化为TF-IDF向量
tv=TfidfVectorizer(stop_words='english',max_features=20000)
tv.fit(train_texts)
voca_dict=tv.vocabulary_
#print(voca_dict) #打印词库

#数据向量化
x_train=tv.fit_transform(train_texts).toarray()
y_train=np.array(train_labels)
x_test=tv.transform(test_texts).toarray()
y_test=np.array(test_labels)

#使用多项式分布的朴素贝叶斯算法训练
classifier=MultinomialNB()
classifier.fit(x_train,y_train)
y_test_predict=classifier.predict(x_test)
y_train_predict=classifier.predict(x_train)

#评估模型的预测效果    
print('The error_rate of train set is',error_rate(y_train_predict,y_train))
print('The error_rate of test set is',error_rate(y_test_predict,y_test)) 
print(classification_report(y_test, y_test_predict,target_names=list(set(y_test))))
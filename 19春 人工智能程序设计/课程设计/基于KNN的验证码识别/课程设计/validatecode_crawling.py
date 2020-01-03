import re
import requests
import time
import random

URL = 'https://www.quanjing.com/createImg.aspx'
for i in range(1000):
    r = requests.get(URL).content
    with open('C:/Users/Jaqen/Desktop/数字训练集/'+str(i+1)+'.jpg', 'wb') as f:
        f.write(r)
    time.sleep(random.random())            
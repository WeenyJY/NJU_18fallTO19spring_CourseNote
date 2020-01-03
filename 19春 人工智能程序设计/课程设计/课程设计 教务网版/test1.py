from PIL import Image
import os
import numpy as np

x=np.array([[1,3,2],[1,4,5]])
y=np.array([[1,2,3],[1,4,6]])
u=list((x.any()==y.any()))
print(u.count(1))
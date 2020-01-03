from PIL import Image
import matplotlib.pyplot as plt
def vertical(img):
    pixel_data = img.load()
    vertical_shadow = []
    for x in range(img.size[0]):
        black = 0
        for y in range(img.size[1]):
            if pixel_data[x,y] == 0:
                black += 1
        vertical_shadow.append(black)
    return vertical_shadow

def get_start_x(vertical_shadow):
#根据图片垂直投影的结果来确定起点,中间值前后取4个值 再这范围内取最小值
    middle = len(vertical_shadow) // 2 
    temp = vertical_shadow[middle-4:middle+5]
    return  middle-4+temp.index(min(temp))

def get_nearby_pix_value(pixel,x,y,j):
#获取临近5个点像素数据
    if j == 1:
        return 0 if pixel[x-1,y+1] == 0 else 1  #左下
    elif j ==2:
        return 0 if pixel[x,y+1] == 0 else 1   #下
    elif j ==3:
        return 0 if pixel[x+1,y+1] == 0 else 1  #右下
    elif j ==4:
        return 0 if pixel[x+1,y] == 0 else 1  #右
    elif j ==5:
        return 0 if pixel[x-1,y] == 0 else 1 #左
    else:
        raise Exception("get_nearby_pix_value error")

def get_end_route(img,start_x):
#获取滴水路径
    left_limit = 0     #左边界
    right_limit = img.size[0] - 1  #右边界
    height=img.size[1]-1         #高度/下边界
    end_route = []    #路径
    cur_p = (start_x,0)   #current_point
    #last_p = cur_p        #last_point
    end_route.append(cur_p)
 
    while cur_p[1] < height:
        sum_n = 0
        max_w = 0         #周围5个点中的势能最大值
        next_x = cur_p[0]
        next_y = cur_p[1]
        pix_img = img.load()      #像素矩阵
        for i in range(1,6):
            cur_w = get_nearby_pix_value(pix_img,cur_p[0],cur_p[1],i) * (6-i)
            sum_n += cur_w
        if max_w < cur_w:
            max_w = cur_w
        if sum_n == 0 or sum_n == 15 :     # 如果全黑或全白则看惯性
            max_w = 4
 
        if max_w == 1:                  #向左
            next_x = cur_p[0] - 1    
            next_y = cur_p[1]
        elif max_w == 2:                  #向右
            next_x = cur_p[0] + 1
            next_y = cur_p[1]
        elif max_w == 3:                   #向右下
            next_x = cur_p[0] + 1
            next_y = cur_p[1] + 1
        elif max_w == 5:                   #向左下
            next_x = cur_p[0] - 1
            next_y = cur_p[1] + 1
        elif max_w == 4:                  #向下
            next_x = cur_p[0]  
            next_y = cur_p[1] + 1
        else:
            raise Exception("get end route error")
 
        if next_x >= right_limit:
            next_x = right_limit-1
        
        if next_x <= left_limit:
            next_x = left_limit+1
        
        cur_p = (next_x,next_y)
        end_route.append(cur_p)

    return end_route


img=Image.open('C:/Users/Jaqen/Desktop/W_0.png')

#startx = get_start_x (vertical(img))
print(vertical(img))
p=vertical(img)
plt.bar(range(img.size[0]),p)
p.remove(10)
img.crop((0,0,p.index(max(p))+1,img.size[1])).show()
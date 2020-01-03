from PIL import Image


def check_letters(letter):       #检查切割点列表，若有大于四个切割处则进行一些操作
    cur_letters=letter         #current_letters
    if len(letter)==4:     
        temp=[x[1]-x[0] for x in letter]
        for i in range(4):
            if temp[i]<4:      #如果切割宽度过短
                cur_letters[i]=(letter[i][0],letter[i][0]+6)  #将宽度设为6
            elif temp[i]>10:   #如果切割宽度过长
                cur_letters[i]=(letter[i][0],letter[i][0]+9)  #将宽度设为9
        return cur_letters,True
    elif len(letter)>4:  #进行一些合并
        final_letters=[]
        temp=[x[1]-x[0] for x in letter]
        i=0
        while i < len(temp):
            if i<len(temp)-1 and temp[i]<4:
                if temp[i+1]<4:
                    final_letters.append((letter[i][0],letter[i+1][1]))  #两个相邻的切割宽度都小于4，则合并
                    i+=2
                else:
                    final_letters.append(letter[i])
                    i+=1
            elif i==len(letter)-1 and temp[i]<4:
                final_letters.remove(final_letters[-1])
                final_letters.append((letter[i-1][0],letter[i][1]))
                i+=1
            else:
                final_letters.append(letter[i])
                i+=1
        cur_letters=final_letters
        return cur_letters,False
    else:
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
            letters.append((start,end))
            inletter=False            
        touchletter=False 
    letters,boolean=check_letters(letters)
    times=1
    while boolean==False and times<3:
        letters,boolean=check_letters(letters)
        times+=1
    return letters
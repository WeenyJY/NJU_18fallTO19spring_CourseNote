import matplotlib.pyplot as plt
import re
import requests
from scipy.misc import imread
from wordcloud import WordCloud

def fetch():
    PATTERN = re.compile('<p><font>(.*?)<\/font><\/p>')
    BASE_URL1 = 'https://cj.sina.com.cn/articles/view/1720289731/668989c301900jk5r?from=finance'
    BASE_URL2='http://k.sina.com.cn/article_1691007117_64cab88d02000kzwz.html?from=news&subch=onews'
    BASE_URL3='http://k.sina.com.cn/article_2288064900_8861198402000kzjr.html?from=edu'
    BASE_URL4='http://k.sina.com.cn/article_6878450622_199fcd3be00100iooo.html?from=edu'
    BASE_URL5='http://k.sina.com.cn/article_6928927807_19cff0c3f00100l043.html?from=edu'
    html_list=[BASE_URL1,BASE_URL2,BASE_URL3,BASE_URL4,BASE_URL5]
    with open('C:/Users/Jaqen/Desktop/news.txt', 'a', encoding='utf-8') as f:
        for i in html_list:
            r = requests.get(i)
            r.encoding='utf-8'
            data = r.text
            p = re.findall(PATTERN, data)
            for s in p:
                f.write(s+'\n')

def extract_words():
    with open('C:/Users/Jaqen/Desktop/news.txt','r', encoding='utf-8') as f:
        news_subjects = f.readlines()   
        stop_words = set(line.strip() for line in open('C:/Users/Jaqen/Desktop/stopwords.txt', encoding='utf-8'))   
        newslist = []
        for subject in news_subjects:
            if subject.isspace():
                continue
            p = re.compile("n[a-z0-9]{0,2}") 
            word_list = pseg.cut(subject)     
            for word, flag in word_list:
                if word not in stop_words and re.match(p,flag)!= None:
                    newslist.append(word)    
        content = {}
        for item in newslist:
            content[item] = content.get(item, 0) + 1
    
    mask_image = imread("C:/Users/Jaqen/Desktop/jiangsu.png")
    wordcloud = WordCloud(font_path=r'C:/Users/Jaqen/Desktop/simhei.ttf', background_color="white", mask=mask_image, max_words=150).generate_from_frequencies(content)
    # Display the generated image:
    plt.imshow(wordcloud)
    plt.axis("off")
    wordcloud.to_file(r'C:/Users/Jaqen/Desktop/wordcloud.jpg')
    plt.show()
    
if __name__ == "__main__":
    fetch()
    extract_words()    






    
    

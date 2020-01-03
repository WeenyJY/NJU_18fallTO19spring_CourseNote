import requests, time
import hmac, json  #这几个需要导入的可以从js中获取，加密需要的
from bs4 import BeautifulSoup
import http.cookiejar as cookielib

session = requests.Session()

def login(username, password,validatecode):     
    data = {
        'Login1_loginName': username,
        'Login1_loginPwd': password,
        'Login1_code': validatecode
        } 
    header = {
       'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
       'accept-Encoding': 'gzip, deflate','br'
       'accept-Language': 'zh-CN,zh;q=0.9',
       'cache-Control': 'max-age=0',
       'upgrade-Insecure-Requests': '1',
       'user-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36 ',
       'cookie': 'ASP.NET_SessionId=neznj3nf5w2z0i22g2t4jpyb; BIGipServerpool-web-ssl=1431087296.47873.0000; qimo_seosource_578c8dc0-6fab-11e8-ab7a-fda8d0606763=%E7%BB%94%E6%AC%8F%E5%94%B4; qimo_seokeywords_578c8dc0-6fab-11e8-ab7a-fda8d0606763=; accessId=578c8dc0-6fab-11e8-ab7a-fda8d0606763; QuanjingHeadShopCartCount_1837071=0; QuanjingHeadShopCartCount_1837071=0; validateCode=89daa89c41ffb71f97a15ad3ae830a06; pageViewNum=28',
       'referer': 'https://www.quanjing.com/'
    }
    print("-" * 100)
    #r = session.get('https://www.quanjing.com/login.aspx',headers=header)
    #cookies_dict=session.cookies.get_dict()
    #print(cookies_dict)
    #items = cookies_dict.items()
    #cookies_lst=[]
    #for item in items:
    #    key,value=item[0],item[1]
    #    cookies_lst.append(key+'='+value)
    #header['Cookie']=';'.join(cookies_lst)
    resp = session.post('https://www.quanjing.com/login.aspx',data=data,headers=header)
    #print(requests.utils.dict_from_cookiejar(resp.cookies))
    #print(header)
#def isLoginStatus():
    routeUrl = "https://www.quanjing.com/commerce/cart.aspx"
    # 下面有两个关键点
    # 第一个是header，如果不设置，会返回500的错误
    # 第二个是allow_redirects，如果不设置，session访问时，服务器返回302，
    # 然后session会自动重定向到登录页面，获取到登录页面之后，变成200的状态码
    # allow_redirects = False  就是不允许重定向
    responseRes = session.get(routeUrl,headers=header,allow_redirects = False)
    print(responseRes.status_code)
    print(responseRes.text)
    #if responseRes.status_code != 200:
    #    print("Failed")#return False
    #else:
    #    print("Successfully login")
    #    #return True
    #
    #return responseRes

if __name__ == "__main__":
    login('NJUAIprogram_2019','2019AIprogramming','7673') # 用户名密码换成自己的
    #print(response.text)

'''from urllib import request
from http import cookiejar
 
if __name__ == '__main__':
    # 声明一个CookieJar对象实例来保存cookie
    cookie = cookiejar.CookieJar()
    # 利用urllib.request库的HTTPCookieProcessor对象来创建cookie处理器,也就CookieHandler
    handler=request.HTTPCookieProcessor(cookie)
    # 通过CookieHandler创建opener
    opener = request.build_opener(handler)
    # 此处的open方法打开网页
    response = opener.open('http://elite.nju.edu.cn/jiaowu/')
    # 打印cookie信息
    for item in cookie:
        print('Name = %s' % item.name)
        print('Value = %s' % item.value)'''
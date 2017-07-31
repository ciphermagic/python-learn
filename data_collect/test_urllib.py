from urllib import request

rep = request.urlopen("http://www.baidu.com")
print(rep.read().decode("utf-8"))
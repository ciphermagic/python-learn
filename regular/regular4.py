import urllib.request
import re
import sys

print(sys.getdefaultencoding())

req = urllib.request.urlopen("http://www.imooc.com/")

html = req.read()

html = html.decode('utf-8')

listurl = re.findall(r"""src\s*=\s*['|"]?.*?jpg['|"]?\s*""", html)

i = 0

for url in listurl:
    url = url[5:-2]
    print(url)
    req = urllib.request.urlretrieve(url, "G:\\新建文件夹\\" + str(i) + ".jpg")
    i = i + 1

import requests
import http.cookiejar as cookielib
import re
import time
from zheye import zheye
import json

session = requests.session()
session.cookies = cookielib.LWPCookieJar(filename="cookies.txt")
try:
    session.cookies.load(ignore_discard=True)
except:
    print("加载cookies失败")

headers = {
    "Host": "www.zhihu.com",
    "Referer": "https://www.zhihu.com/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36",
}


def get_captcha():
    t = str(int(time.time() * 1000))
    captcha_url = "https://www.zhihu.com/captcha.gif?r={0}&type=login&lang=cn".format(t)
    t = session.get(captcha_url, headers=headers)
    with open("captcha.jpg", "wb") as f:
        f.write(t.content)
        f.close()
    z = zheye()
    pos = z.Recognize("captcha.jpg")
    tmp = []
    input_points = []
    for poss in pos:
        tmp.append(float(format(poss[1] / 2, '0.2f')))
        tmp.append(float(format(poss[0] / 2, '0.2f')))
        input_points.append(tmp)
        tmp = []
    result = '{"img_size": [200, 44], "input_points": %s}' % str(input_points)
    return result


def is_login():
    inbox_url = "https://www.zhihu.com/inbox"
    response = session.get(inbox_url, headers=headers, allow_redirects=False)
    if response.status_code != 200:
        return False
    else:
        return True


def get_index():
    response = session.get("https://www.zhihu.com", headers=headers)
    with open("index_page.html", "wb") as f:
        f.write(response.text.encode("utf-8"))
        f.close()
    print("ok")


def get_xsrf():
    response = session.get("https://www.zhihu.com", headers=headers)
    text = response.text
    match_obj = re.search('.*name="_xsrf"\s+value="(.*?)"', text)
    if match_obj:
        return match_obj.group(1)
    else:
        return ""


def zhihu_login(account, password):
    if re.match("1\d{10}", account):
        print("手机登录")
        post_url = "https://www.zhihu.com/login/phone_num"
        post_data = {
            "_xsrf": get_xsrf(),
            "phone_num": account,
            "password": password,
            "captcha": get_captcha(),
            "captcha_type": "cn",
        }
    elif "@" in account:
        print("邮箱登陆")
        post_url = "https://www.zhihu.com/login/email"
        post_data = {
            "_xsrf": get_xsrf(),
            "email": account,
            "password": password,
            "captcha": get_captcha(),
            "captcha_type": "cn",
        }
    response = session.post(post_url, data=post_data, headers=headers)
    session.cookies.save()
    print(json.loads(response.text))


if __name__ == "__main__":
    if is_login():
        print("登陆成功")
    else:
        zhihu_login("ciphermagic@yeah.net", "XXX")
        print(is_login())

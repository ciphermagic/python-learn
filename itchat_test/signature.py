import itchat
import re
import jieba
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator
import PIL.Image as Image
import os
import numpy as np

# 先登录
itchat.auto_login(hotReload=True)

# 获取好友列表
friends = itchat.get_friends(update=True)[0:]
tList = []

for i in friends:
    # 正则匹配过滤掉emoji表情，例如emoji1f3c3等
    signature = i["Signature"].strip().replace("span", "").replace("class", "").replace("emoji", "")
    rep = re.compile("<.*>")
    signature = rep.sub("", signature)
    tList.append(signature)
    # 拼接字符串
    text = "".join(tList)
    # jieba分词
    wordlist_jieba = jieba.cut(text, cut_all=True)
    wl_space_split = " ".join(wordlist_jieba)

# 这里要选择字体存放路径，这里是Mac的，win的字体在windows／Fonts中
d = os.path.dirname(__file__)
alice_coloring = np.array(Image.open(os.path.join(d, "wechat.png")))
my_wordcloud = WordCloud(background_color="white",
                         max_words=2000,
                         max_font_size=40,
                         random_state=42,
                         mask=alice_coloring,
                         font_path='C:\Windows\Fonts\msyh.ttc').generate(wl_space_split)
image_colors = ImageColorGenerator(alice_coloring)
# plt.imshow(my_wordcloud)
plt.imshow(my_wordcloud.recolor(color_func=image_colors))
plt.axis("off")
plt.show()
# 保存图片 并发送到手机
my_wordcloud.to_file(os.path.join(d, "wechat_cloud.png"))
itchat.send_image("wechat_cloud.png", 'filehelper')

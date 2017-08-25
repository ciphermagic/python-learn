from selenium import webdriver
import time
import urllib.request
from bs4 import BeautifulSoup
import html.parser


def main():
    chromedriver = "C:\\Users\\cipher\\AppData\\Local\\Google\\Chrome\\Application\\chromedriver.exe"
    driver = webdriver.Chrome(chromedriver)
    # 列出来你想要下载图片的网站
    # driver.get("https://www.zhihu.com/question/35931586") # 你的日常搭配是什么样子？
    # driver.get("https://www.zhihu.com/question/61235373") # 女生腿好看胸平是一种什么体验？
    # driver.get("https://www.zhihu.com/question/28481779") # 腿长是一种什么体验？
    # driver.get("https://www.zhihu.com/question/19671417") # 拍照时怎样摆姿势好看？
    # driver.get("https://www.zhihu.com/question/20196263") # 女性胸部过大会有哪些困扰与不便？
    # driver.get("https://www.zhihu.com/question/46458423") # 短发女孩要怎么拍照才性感？
    driver.get("https://www.zhihu.com/question/26037846")  # 身材好是一种怎样的体验？

    # **************** 滑动到页面底部，点击加载更多按钮 ****************
    def execute_times(times):
        for i in range(times):
            # 滑动到浏览器底部
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            # 等待页面加载
            time.sleep(2)
            try:
                # 选中并点击页面底部的加载更多
                driver.find_element_by_css_selector("button.QuestionMainAction").click()
                # 输出页面页数
                print("page" + str(i))
                time.sleep(1)
            except:
                break

    # **************** 查找所有 <nonscript> 节点并保存图片HTML信息****************
    def parserImg():
        # 原页面
        result_raw = driver.page_source
        # 解析页面
        result_soup = BeautifulSoup(result_raw, "html.parser")
        with open("./output/rawfile/noscript_meta.txt", 'wb') as noscript_meta:
            # 找到所有 <noscript> 节点
            noscript_nodes = result_soup.find_all('noscript')
            noscript_inner_all = ""
            for noscript in noscript_nodes:
                # 获取<noscript>node内部内容
                noscript_inner = noscript.get_text()
                noscript_inner_all += noscript_inner + "\n"
            # 将内部内容转码并存储
            noscript_all = html.parser.unescape(noscript_inner_all)
            noscript_meta.write(noscript_all.encode("utf-8"))
            noscript_meta.close()
        print("Store noscript meta data successfully!")
        return noscript_all

    # **************** 保存图片 URL 和内容 ****************
    def saveImg():
        noscript_all = parserImg()
        img_soup = BeautifulSoup(noscript_all, 'html.parser')
        img_nodes = img_soup.find_all('img')
        with open("./output/rawfile/img_meta.txt", 'wb') as img_meta:
            count = 0
            for img in img_nodes:
                img_url = img.get('src')
                if img_url is not None:
                    line = str(count) + "\t" + img_url + "\n"
                    img_meta.write(line.encode("utf-8"))
                    # 一个一个下载图片
                    urllib.request.urlretrieve(img_url, "./output/image/" + str(count) + ".jpg")
                    count += 1
            img_meta.close()
        print("Store meta data and images successfully!")

    execute_times(5)
    saveImg()


if __name__ == '__main__':
    main()

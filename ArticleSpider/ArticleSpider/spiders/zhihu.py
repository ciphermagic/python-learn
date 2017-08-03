# -*- coding: utf-8 -*-
import datetime
import scrapy
from ArticleSpider import settings
import re
import json
import time
from zheye import zheye
from urllib import parse
from ArticleSpider.items import ZhihuAnswerItem, ZhihuQuestionItem, ArticleItemLoader


class ZhihuSpider(scrapy.Spider):
    name = "zhihu"
    allowed_domains = ["www.zhihu.com"]
    start_urls = ['https://www.zhihu.com/']

    def start_requests(self):
        yield scrapy.Request(
            url="https://www.zhihu.com/#signin",
            callback=self.login,
        )

    def login(self, response):
        text = response.text
        match_obj = re.match('.*name="_xsrf"\s+value="(.*?)"', text, re.DOTALL)
        xsrf = ""
        if match_obj:
            xsrf = match_obj.group(1)
        if xsrf:
            post_data = {
                "_xsrf": xsrf,
                "email": settings.ZHIHU_USER,
                "password": settings.ZHIHU_PASSWD,
                "captcha_type": "cn",
            }
            t = str(int(time.time() * 1000))
            captcha_url = "https://www.zhihu.com/captcha.gif?r={0}&type=login&lang=cn".format(t)
            yield scrapy.Request(
                url=captcha_url,
                callback=self.login_after_captcha,
                meta={"post_data": post_data}
            )

    def login_after_captcha(self, response):
        with open("captcha.jpg", "wb") as f:
            f.write(response.body)
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
        captcha = '{"img_size": [200, 44], "input_points": %s}' % str(input_points)
        post_data = response.meta.get("post_data", {})
        post_data["captcha"] = captcha
        yield scrapy.FormRequest(
            url="https://www.zhihu.com/login/email",
            formdata=post_data,
            callback=self.check_login
        )

    def check_login(self, response):
        text_json = json.loads(response.text)
        print(text_json)
        if "msg" in text_json and text_json["msg"] == "登录成功":
            print("登录成功")
            for url in self.start_urls:
                yield scrapy.Request(url=url, dont_filter=True)
        else:
            print("登录失败")

    def parse(self, response):
        all_urls = response.css("a::attr(href)").extract()
        all_urls = [parse.urljoin(response.url, url) for url in all_urls]
        all_urls = filter(lambda x: True if x.startswith("https") else False, all_urls)
        for url in all_urls:
            match_obj = re.match("(.*zhihu.com/question/(\d+))(/|$).*", url)
            if match_obj:
                request_url = match_obj.group(1)
                question_id = int(match_obj.group(2))
                yield scrapy.Request(
                    url=request_url,
                    meta={"question_id": question_id},
                    callback=self.parse_question
                )

    @staticmethod
    def parse_question(response):
        item_loader = ArticleItemLoader(item=ZhihuQuestionItem(), response=response)
        item_loader.add_css("title", "h1.QuestionHeader-title::text")
        item_loader.add_css("content", ".QuestionHeader-detail")
        item_loader.add_value("url", response.url)
        item_loader.add_value("zhihu_id", response.meta.get("question_id", []))
        item_loader.add_css("answer_num", ".List-headerText span::text")
        item_loader.add_css("comments_num", ".QuestionHeader-Comment button::text")
        item_loader.add_css("watch_user_num", "button.NumberBoard-item .NumberBoard-value::text")
        item_loader.add_css("click_num", "div.NumberBoard-item .NumberBoard-value::text")
        item_loader.add_css("topics", ".QuestionHeader-topics .Popover div::text")
        item_loader.add_value("crawl_time", datetime.datetime.now())
        question_item = item_loader.load_item()
        yield question_item

    def parse_answer(self, reponse):
        # 处理question的answer
        ans_json = json.loads(reponse.text)
        is_end = ans_json["paging"]["is_end"]
        next_url = ans_json["paging"]["next"]
        # 提取answer的具体字段
        for answer in ans_json["data"]:
            answer_item = ZhihuAnswerItem()
            answer_item["zhihu_id"] = answer["id"]
            answer_item["url"] = answer["url"]
            answer_item["question_id"] = answer["question"]["id"]
            answer_item["author_id"] = answer["author"]["id"] if "id" in answer["author"] else None
            answer_item["content"] = answer["content"] if "content" in answer else None
            answer_item["parise_num"] = answer["voteup_count"]
            answer_item["comments_num"] = answer["comment_count"]
            answer_item["create_time"] = answer["created_time"]
            answer_item["update_time"] = answer["updated_time"]
            answer_item["crawl_time"] = datetime.datetime.now()
            yield answer_item
        if not is_end:
            yield scrapy.Request(next_url, callback=self.parse_answer)

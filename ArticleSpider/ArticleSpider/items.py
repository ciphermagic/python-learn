# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html
import re

import scrapy
import datetime
from scrapy.loader import ItemLoader
from scrapy.loader.processors import MapCompose, TakeFirst, Join
from ArticleSpider import settings


# 日期转换
def date_convert(value):
    try:
        create_date = datetime.datetime.strptime(value.replace("·", "").strip(), '%Y/%m/%d').date()
    except Exception as e:
        create_date = datetime.datetime.now().date()
    return create_date


# 获取字符串中的数字
def get_nums(value):
    match_re = re.match(".*?(\d+).*", value)
    if match_re:
        nums = int(match_re.group(1))
    else:
        nums = 0
    return nums


# 直接返回值
def return_value(value):
    return value


# 自定义ItemLoader，指定默认的后置过滤器
class ArticleItemLoader(ItemLoader):
    default_output_processor = TakeFirst()


# 自定义Join，过滤某个标签
class TagsJoin(object):
    def __init__(self, separator=u' ', exclude=""):
        self.separator = separator
        self.exclude = exclude

    def __call__(self, values):
        values = [element for element in values if not element.strip().endswith(self.exclude)]
        return self.separator.join(values)


# 伯乐文章Item
class JobBoleArticleItem(scrapy.Item):
    title = scrapy.Field()
    url = scrapy.Field()
    url_object_id = scrapy.Field()
    front_image_path = scrapy.Field()
    content = scrapy.Field()
    front_image_url = scrapy.Field(
        output_processor=MapCompose(return_value)
    )
    create_date = scrapy.Field(
        input_processor=MapCompose(date_convert)
    )
    praise_nums = scrapy.Field(
        input_processor=MapCompose(get_nums)
    )
    fav_nums = scrapy.Field(
        input_processor=MapCompose(get_nums)
    )
    comment_nums = scrapy.Field(
        input_processor=MapCompose(get_nums)
    )
    tags = scrapy.Field(
        output_processor=TagsJoin(",", "评论")
    )

    def get_insert_sql(self):
        front_image_url = self["front_image_url"][0] if self["front_image_url"] else ""
        params = (
            self["url_object_id"],
            self["title"],
            self["url"],
            self["create_date"],
            self["fav_nums"],
            front_image_url,
            self["front_image_path"],
            self["praise_nums"],
            self["comment_nums"],
            self["tags"],
            self["content"]
        )
        return jobbole_sql, params


# 知乎问题Item
class ZhihuQuestionItem(scrapy.Item):
    zhihu_id = scrapy.Field()
    url = scrapy.Field()
    crawl_time = scrapy.Field()
    topics = scrapy.Field(
        output_processor=Join(",")
    )
    title = scrapy.Field(
        output_processor=Join("")
    )
    content = scrapy.Field(
        output_processor=Join("")
    )
    answer_num = scrapy.Field(
        input_processor=MapCompose(get_nums)
    )
    comments_num = scrapy.Field(
        input_processor=MapCompose(get_nums)
    )
    watch_user_num = scrapy.Field(
        input_processor=MapCompose(get_nums)
    )
    click_num = scrapy.Field(
        input_processor=MapCompose(get_nums)
    )

    def get_insert_sql(self):
        params = (
            self["zhihu_id"],
            self["topics"],
            self["url"],
            self["title"],
            self["content"],
            self["answer_num"],
            self["comments_num"],
            self["watch_user_num"],
            self["click_num"],
            self["crawl_time"].strftime(settings.SQL_DATETIME_FORMAT),
        )
        return zhihu_question_sql, params


# 知乎回答Item
class ZhihuAnswerItem(scrapy.Item):
    zhihu_id = scrapy.Field()
    url = scrapy.Field()
    question_id = scrapy.Field()
    author_id = scrapy.Field()
    content = scrapy.Field()
    parise_num = scrapy.Field()
    comments_num = scrapy.Field()
    create_time = scrapy.Field()
    update_time = scrapy.Field()
    crawl_time = scrapy.Field()

    def get_insert_sql(self):
        create_time = datetime.datetime.fromtimestamp(self["create_time"]).strftime(settings.SQL_DATETIME_FORMAT)
        update_time = datetime.datetime.fromtimestamp(self["update_time"]).strftime(settings.SQL_DATETIME_FORMAT)
        params = (
            self["zhihu_id"],
            self["url"],
            self["question_id"],
            self["author_id"],
            self["content"],
            self["parise_num"],
            self["comments_num"],
            create_time,
            update_time,
            self["crawl_time"].strftime(settings.SQL_DATETIME_FORMAT),
        )
        return zhihu_answer_sql, params


# 伯乐文章新增SQL
jobbole_sql = """
    INSERT INTO jobbole_article(
        url_object_id, 
        title, 
        url, 
        create_date, 
        fav_nums, 
        front_image_url, 
        front_image_path,
        praise_nums, 
        comment_nums, 
        tags, 
        content
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) 
    ON DUPLICATE KEY UPDATE 
        content=VALUES(content), 
        comment_nums=VALUES(comment_nums), 
        fav_nums=VALUES(fav_nums), 
        praise_nums=VALUES(praise_nums)
"""

# 知乎问题新增SQL
zhihu_question_sql = """
    INSERT INTO zhihu_question(
        zhihu_id, 
        topics, 
        url, 
        title, 
        content, 
        answer_num, 
        comments_num,
        watch_user_num, 
        click_num, 
        crawl_time
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE 
        content=VALUES(content), 
        answer_num=VALUES(answer_num), 
        comments_num=VALUES(comments_num),
        watch_user_num=VALUES(watch_user_num), 
        click_num=VALUES(click_num)
"""

# 知乎回答新增SQL
zhihu_answer_sql = """
    INSERT INTO zhihu_answer(
        zhihu_id, 
        url, 
        question_id, 
        author_id, 
        content, 
        parise_num, 
        comments_num,
        create_time, 
        update_time, 
        crawl_time
      ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
      ON DUPLICATE KEY UPDATE 
        content=VALUES(content), 
        comments_num=VALUES(comments_num), 
        parise_num=VALUES(parise_num),
        update_time=VALUES(update_time)
"""

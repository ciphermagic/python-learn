# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html


class DingdianPipeline(object):
    def process_item(self, item, spider):
        content = (
                  item['name_id'] + ', ' + item['author'] + ', ' + item['name'] + ', ' + item['category'] + ', ' + item[
                      'novelurl']).replace(u'\xa0', u' ')
        fp = open('text.txt', 'a')
        fp.write(content)
        fp.write('\n')
        fp.close()
        return item

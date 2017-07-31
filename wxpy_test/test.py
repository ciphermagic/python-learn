from wxpy import *

bot = Bot(cache_path=True)
my_friend = bot.friends().search('心心芯some')[0]
my_group = bot.groups().search('我们这一家')[0]
tuling = Tuling(api_key='46d97c0cf4fa40ed84f962eac512497c')


@bot.register(my_group, except_self=False)
def reply_my_friend(msg):
    print(tuling.do_reply(msg))


embed()

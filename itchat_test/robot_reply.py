import requests
import itchat

KEY = '46d97c0cf4fa40ed84f962eac512497c'


def get_response(msg):
    apiUrl = 'http://www.tuling123.com/openapi/api'
    data = {
        'key': KEY,
        'info': msg,
        'userid': 'cipher',
    }
    try:
        r = requests.post(apiUrl, data=data).json()
        return r.get('text')
    except:
        return


@itchat.msg_register(itchat.content.TEXT)
def tuling_reply(msg):
    default_reply = '我收到了: ' + msg['Text']
    reply = get_response(msg['Text'])
    return reply or default_reply


itchat.auto_login(hotReload=True)
itchat.run()

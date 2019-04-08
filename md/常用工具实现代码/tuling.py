# coding = utf8
import requests
import itchat
# 去图灵机器人官网注册后会生成一个apikey，可在个人中心查看
KEY = '227de2ea7ace421994e2724823051328'
def get_response(msg):
    apiUrl = 'http://www.tuling123.com/openapi/api'
    data = {
        'key'   : KEY,
        'info'   : msg,   # 这是要发送出去的信息
        'userid'  : 'wechat-rebot',  #这里随意写点什么都行
    }
    try:
        # 发送一个post请求
        r = requests.post(apiUrl, data =data).json()
        # 获取文本信息，若没有‘Text’ 值，将返回Nonoe 
        return r.get('text')
    except:
        return

@itchat.msg_register(itchat.content.TEXT)
def tuling_reply1(msg):

    if msg['User']['RemarkName'] == '国服刘备备的家养贴脸香' or msg['User']['NickName'] == '辛然':
        username = msg['FromUserName']
        print('-+-+' * 5)
        reply = get_response(msg['Text'])
        print('Message content:%s' % msg['Content'])
        print('My Reply:%s' % (reply))
        print('-+-+' * 5)
        itchat.send('墩儿5号：%s' % reply, username)

# @itchat.msg_register(itchat.content.TEXT, isGroupChat = True)
# def tuling_reply2(msg):

#     if msg['User']['NickName'] == '3点钟婚恋交流群':
#         username = msg['FromUserName']
#         print('-+-+' * 5)
#         reply = get_response(msg['Text'])
#         print('Message content:%s' % msg['Content'])
#         print('My Reply:%s' % (reply))
#         print('-+-+' * 5)
#         itchat.send('智障ai：%s' % reply, username)

# 使用热启动，不需要多次扫码
itchat.auto_login(enableCmdQR=True, hotReload=True)
itchat.run()

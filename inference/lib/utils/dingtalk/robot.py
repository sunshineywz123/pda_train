import os
try: import requests
except: os.system('pip install requests'); import requests
try: from dingtalkchatbot.chatbot import DingtalkChatbot, ActionCard, CardItem
except: os.system('pip install DingtalkChatbot'); from dingtalkchatbot.chatbot import DingtalkChatbot, ActionCard, CardItem
# webhook = 'https://oapi.dingtalk.com/robot/send?access_token=56e7dc92fddd78bbc4d457a08d037e6d866664fd90ddf7100ee3a03882f07153'
# secret = 'SEC1f896436f9c0fffa73a95d973d65f68d0dc9092a6c6e4d016b44a8175bf61c18'
# https://oapi.dingtalk.com/robot/send?access_token=caac9d28c298db0d53a89512b6fcc173fda4e371087540cc937c2f17a711b242
# curl 'https://oapi.dingtalk.com/robot/send?access_token=caac9d28c298db0d53a89512b6fcc173fda4e371087540cc937c2f17a711b242'  -H 'Content-Type: application/json' -d '{"msgtype": "text","text": {"content":"????????????, ?????????????????????"}}'
webhook = 'https://oapi.dingtalk.com/robot/send?access_token=caac9d28c298db0d53a89512b6fcc173fda4e371087540cc937c2f17a711b242'
secret = 'SECec6827303ee1635844d871d118c8a99f96b2955b6f6181d20ea8e82db39b2472'
robot = DingtalkChatbot(webhook, secret=secret)
message='hi 111'
robot.send_text(msg=message)
def send_message(message):
    try: robot.send_text(msg=message)
    except: print('Failed to send message')
if __name__ == '__main__':
    send_message('hi')
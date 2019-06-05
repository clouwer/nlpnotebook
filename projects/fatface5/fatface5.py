"""
    Group Chat Robot v0.1
"""
# coding: utf-8

import itchat, re
from itchat.content import *
from random import choice
import json
import configparser
import os

def intend_str_list_load_one(intend):
	list_path = os.path.join("./chat_sentence/sentence_", intend, '.txt')
	str_list = ''
	with open(list_path, "r",encoding='UTF-8') as f:
	    str_list = f.readlines()
	return choice(str_list)

@itchat.msg_register([TEXT], isFriendChat= True)
def text_reply(msg):

	# print(msg['User'])
	if msg['User']['RemarkName'] == my_lady_wechat_name:
		username = msg['FromUserName']

	####################################################
	#                                                  #
	#               intend classification              #
	#                                                  #
	####################################################
	if intent_class == 'say_love':
		itchat.send('{}'.format(love_message), username)

if __name__ == "__main__":
	# 读取配置文件
	conf = configparser.ConfigParser()
	conf.read("./config.ini",encoding='UTF-8')

	my_lady_wechat_name = conf.get("configuration", "my_lady_wechat_nick_name")

	intend_list = conf.get("intend classification", "intend_list")

	intend_reply = {}
	for intend in intend_list:
		intend_reply[intend] = intend_str_list_load_one(intend)

	itchat.auto_login(enableCmdQR=True, hotReload=True)
	itchat.run()
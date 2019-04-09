
import json
import os
import time

class Config(object):

	"""Docstring for Config. """

	def __init__(self):

		self.origin_data_file = 'C:\\Jupyter\\博圣工作\\其他服务项\\维修项目\\service\\data\\2015-2018维修数据_部分.xlsx'
		self.savefig_path = '../source/fig/'
		os.makedirs(self.savefig_path, exist_ok=True)
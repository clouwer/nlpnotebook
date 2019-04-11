
import json
import os
import time

class Config(object):

	"""Docstring for Config. """

	def __init__(self):

		self.origin_data_filepath = 'C:\\Jupyter\\博圣工作\\其他服务项\\维修项目\\service\\data\\'
		self.savefig_path = '../source/fig/'
		os.makedirs(self.savefig_path, exist_ok=True)

		self.sql_user = 'root'
		self.sql_password = 'mysql123'
		self.sql_ip = '172.16.0.164'
		self.sql_database = 'service_upgrade_xr'
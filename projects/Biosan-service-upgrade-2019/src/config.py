
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

		self.userdict_file = '../source/words/problem.txt'
		self.seg_senquence_file = '../source/words/jieba_result.txt'

		self.savemodel_path = '../source/model/'
		os.makedirs(self.savemodel_path, exist_ok=True)
		self.embed_size = 300
		self.window_size = 10
		self.w2v_file = '../source/model/word2vec.{}d.wsize{}.model'.format(self.embed_size, self.window_size)

		self.savenlp_path = '../source/words/'
		os.makedirs(self.savenlp_path, exist_ok=True)

		self.lgb_max_params = {'colsample_bytree': 0.5,
								 'min_child_samples': 7.347950100676255,
								 'num_leaves': 96.09357631764188,
								 'reg_alpha': 1.5571088120843582e-06,
								 'reg_lambda': 1.2860202034063392e-07,
								 'subsample': 0.5}
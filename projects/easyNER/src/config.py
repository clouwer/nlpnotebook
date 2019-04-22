
import json
import os
import time

class Config(object):

	"""Docstring for Config. """

	def __init__(self):

		self.origin_datapath = '../source/data/origin_data/'
		self.userdict = '../source/data/ner_data/ner_pos.txt'

		self.label_dict = {'检查和检验': 'CHECK',
			              '症状和体征': 'SIGNS',
			              '疾病和诊断': 'DISEASE',
			              '治疗': 'TREATMENT',
			              '身体部位': 'BODY'}
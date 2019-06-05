
import json
import os
import time

class Config(object):

	"""Docstring for Config. """

	def __init__(self):

		cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
		self.origin_datapath = '../source/data/origin_data/'
		self.userdict = '../source/data/ner_data/ner_pos.txt'
		self.train_path = '../source/data/ner_data/ner_train.txt'
		self.vocab_path = os.path.join(cur, 'model/vocab.txt')

		self.label_dict = {'检查和检验': 'CHECK',
			              '症状和体征': 'SIGNS',
			              '疾病和诊断': 'DISEASE',
			              '治疗': 'TREATMENT',
			              '身体部位': 'BODY'}

		self.class_dict ={
	                         'O':0,
	                         'B-TREATMENT': 1,
	                         'I-TREATMENT': 2,
	                         'B-BODY': 3,
	                         'I-BODY': 4,
	                         'B-SIGNS': 5,
	                         'I-SIGNS': 6,
	                         'B-CHECK': 7,
	                         'I-CHECK': 8,
	                         'B-DISEASE': 9,
	                         'I-DISEASE': 10
	                        }
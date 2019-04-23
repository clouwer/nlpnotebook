
# coding: utf-8

from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.seq_labeling import BLSTMCRFModel
from config import Config
import random
import os

class BERTNER:
	def __init__(self):
		cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
		self.train_path = os.path.join(cur, '../source/data/ner_data/ner_train.txt')
		self.model_path = os.path.join(cur, 'model/bert_model_20.h5')
		self.vocab_path = os.path.join(cur, 'model/vocab.txt')
		self.datas, self.word_dict = self.build_data()
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
		self.EPOCHS = 100
		self.BATCH_SIZE = 500
		self.embedding = BERTEmbedding('bert-base-chinese', 100)

	def write_file(self, wordlist, filepath):
		with open(filepath, 'w+', encoding = 'UTF-8') as f:
			f.write('\n'.join(wordlist))

	def build_data(self):
		datas = []
		sample_x = []
		sample_y = []
		vocabs = {'UNK'}
		for line in open(config.train_path, encoding = 'UTF-8'):
			line = line.rstrip().split('\t')
			if not line:
				continue
			char = line[0]
			if not char:
				continue
			cate = line[-1]
			sample_x.append(char)
			sample_y.append(cate)
			vocabs.add(char)
			if char in ['。','?','!','！','？']:
				datas.append([sample_x, sample_y])
				sample_x = []
				sample_y = []
		word_dict = {wd:index for index, wd in enumerate(list(vocabs))}
		self.write_file(list(vocabs), config.vocab_path)
		return datas, word_dict

	def data_load(self, validation_split= 0.2):

		datas, word_dict = self.build_data()
		random.shuffle(datas)
		x = [[char for char in data[0]] for data in datas]
		y = [[label for label in data[1]] for data in datas]

		x_train = x[:int(len(x)*validation_split)]
		y_train = y[:int(len(y)*validation_split)]
		x_valid = x[int(len(x)*validation_split)+1:]
		y_valid = y[int(len(y)*validation_split)+1:]
		return x_train, y_train, x_valid, y_valid

	# 还可以选择 `BLSTMModel` 和 `CNNLSTMModel` 

	def train_model(self):
		x_train, y_train, x_valid, y_valid = self.data_load(validation_split= 0.2)
		model = BLSTMCRFModel(self.embedding)
		model.fit(x_train, y_train, x_validate=x_valid, y_validate=y_valid,
		          epochs= self.EPOCHS,  batch_size= self.BATCH_SIZE)
		model.save(self.model_path)

	def build_input(self, text):
	    datas = []
	    x = []
	    for char in text:
	        x.append(char)
	        if char in ['。','?','!','！','？'] or text.index(char) == len(text)-1:
	            datas.append(x)
	            x = []
	    return datas

	def model_predict(self, model, text):
		x_test = self.build_input(text)
		result = model.predict(x_test)
		chars = [i for i in text]
		tags = []
		for i in range(len(result)):
			tags = tags + result[i]
		res = list(zip(chars, tags))
		print(res)
		return(res)

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--option', type = str, default = 'predict')
	args = parser.parse_args()
	config = Config()

	bertner = BERTNER()
	if args.option == 'train':
		bertner.train_model()
	else:
		model = BLSTMCRFModel.load_model(bertner.model_path)
		while 1:
			s = input('enter an sent:').strip()
			bertner.model_predict(model, s)

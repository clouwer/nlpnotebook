
# coding: utf-8

from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.seq_labeling import BLSTMCRFModel
from config import Config

def build_data():
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
    def write_file(wordlist, filepath):
        with open(filepath, 'w+', encoding = 'UTF-8') as f:
            f.write('\n'.join(wordlist))
    write_file(list(vocabs), config.vocab_path)
    return datas, word_dict

def data_load(validation_split= 0.2):

	datas, word_dict = build_data()
	random.shuffle(datas)
	x = [[char for char in data[0]] for data in datas]
	y = [[label for label in data[1]] for data in datas]

	x_train = x[:int(len(x)*validation_split)]
	y_train = y[:int(len(y)*validation_split)]
	x_valid = x[int(len(x)*validation_split)+1:]
	y_valid = y[int(len(y)*validation_split)+1:]
	return x_train, y_train, x_valid, y_valid

# 还可以选择 `BLSTMModel` 和 `CNNLSTMModel` 

def train_model(embedding):
	x_train, y_train, x_valid, y_valid = data_load(validation_split= 0.2)
	model = BLSTMCRFModel(embedding)
	model.fit(x_train, y_train, x_validate=x_valid, y_validate=y_valid,
	          epochs= 20,  batch_size=500)
	model.save('model/bert_model_20.h5')

def build_input(text):
    datas = []
    x = []
    for char in text:
    	x.append(char)
		if char in ['。','?','!','！','？']:
		    datas.append(x)
    return datas

if __name__ == '__main__':
	import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type = str, default = 'predict')
    args = parser.parse_args()
	config = Config()
	embedding = BERTEmbedding('bert-base-chinese', 100)

	if args.option == 'train':
	    train_model(embedding)
	else:
		while 1:
			model = BLSTMCRFModel(embedding)
			model.load_weights('model/bert_model_20.h5')
	        s = input('enter an sent:').strip()
	        x_test = build_input(s)
	        print(model.predict(x_test))

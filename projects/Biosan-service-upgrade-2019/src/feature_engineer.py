
import pandas as pd
import numpy as np
import warnings
import jieba
import jieba.posseg as pseg
jieba.load_userdict('../source/words/problem.txt')
import re
from gensim.models import Word2Vec
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sqlalchemy import create_engine
from config import *


def data_sql_load():
	engine = create_engine('mysql+pymysql://%s:%s@%s:3306/%s?charset=utf8' %(config.sql_user, config.sql_password, config.sql_ip, config.sql_database),echo = False)
	sql = "select * from etl_data"
	data = pd.read_sql_query(sql, engine)
	return data

# 创建停用词list  
def stopwordslist(filepath):  
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]  
    return stopwords    
  
# 对句子进行分词  
def seg_sentence(sentence):  
    sentence_seged = jieba.cut(sentence.strip(), HMM=True, cut_all=False)  
#     sentence_seged = jieba.cut_for_search(sentence.strip(), HMM=True)
    stopwords = stopwordslist('../source/words/中文停用词表.txt')  # 这里加载停用词的路径
    outstr = ''  
    for word in sentence_seged:  
        if word not in stopwords:  
            if word != '\t':  
                outstr += word  
                outstr += " "  
    return outstr

# jieba分词后文本转词向量
def w2v_split(dataseries, window_size = 3, embed_size = 300): 
    w2v_col = [f'w2v_{i}' for i in range(embed_size)]
    all_texts = []
    dataseries.fillna('',inplace = True)
    for seq in dataseries:
        for shift in range(0,window_size):
            all_texts.append([word for word in re.findall(r'.{'+str(window_size)+'}',seq[shift:])])
    model = Word2Vec(all_texts,size=embed_size,window=4,min_count=1,negative=3,
                     sg=1,sample=0.001,hs=1,workers=4,iter=15)
    w2v_feat = []
    i = 0
    while i <= len(all_texts)-window_size:
        sum_w2v = np.zeros(shape=(embed_size,))
        for j in range(i,i+window_size):
            for word in all_texts[j]:
                sum_w2v += model[word]
        w2v_feat.append(sum_w2v)
        i = i+window_size
    w2v_feat = np.vstack(w2v_feat)
    df_w2v = pd.DataFrame(w2v_feat,columns=w2v_col)
    return df_w2v

def lgb_w2v_train(X, Y, max_params, validation_size = 0.3):
    X_model, X_pred, Y_model, Y_pred = train_test_split(X, Y, test_size = validation_size,
                                                        random_state= 2019)
    train = lgb.Dataset(X_model, label=Y_model)
    valid = train.create_valid(X_pred, label=Y_pred)
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'min_child_samples': int(max_params['min_child_samples']),
        'num_leaves': int(max_params['num_leaves']),
        'reg_alpha': max_params['reg_alpha'],
        'reg_lambda': max_params['reg_lambda'],
        'subsample': max_params['subsample'],
        'colsample_bytree': max_params['colsample_bytree'],
        'learning_rate': 0.05,
        'seed': 2019,
        'nthread': -1
    }
    num_round = 40000
    gbm = lgb.train(params, train, num_round,
                    verbose_eval=500,valid_sets=[train, valid],early_stopping_rounds= 1000)
    return gbm

def lgb_bayesoptimization(X, y, init_points = 10, num_iter = 25,random_state = 2019):
	def lgb_evaluate(num_leaves, subsample, colsample_bytree, min_child_samples, reg_alpha, reg_lambda):

	    train = lgb.Dataset(X, label= y)
	    params = {
	        'boosting_type': 'gbdt',
	        'objective': 'binary',
	        'metric': 'auc',
	        'min_child_samples': int(min_child_samples),
	        'num_leaves': int(num_leaves),
	        'reg_alpha': max(reg_alpha,0),
	        'reg_lambda': max(reg_lambda,0),
	        'subsample': max(min(subsample, 1), 0),
	        'colsample_bytree': max(min(colsample_bytree, 1), 0) ,
	        'learning_rate': 0.05,
	        'seed': 2019,
	        'nthread': -1,}
	    cv_result = lgb.cv(params, train, nfold = 5, seed= 2019, stratified=True, verbose_eval =200, metrics=['auc'])
	    return max(cv_result['auc-mean'])
	lgbBO = BayesianOptimization(lgb_evaluate, {'num_leaves': (60, 120), 
	                                            'subsample': (0.5, 1),
	                                            'colsample_bytree': (0.5, 1),
	                                            'min_child_samples': (5,100), 
	                                            'reg_alpha': (0,10),
	                                            'reg_lambda': (0,10)
	                                            })
	lgbBO.maximize(init_points=init_points, n_iter=num_iter)
	return lgbBO

def lgb_w2v_train(X, y):
	w2v_train = w2v_split(X)
	lgbBO = lgb_bayesoptimization(w2v_train, y)
	max_params = lgbBO.res['max']['max_params']
	lgb_model = lgb_w2v_train(w2v_train, y, max_params)
	return lgb_model

def feature_engineer():

	data = data_sql_load()
	data['问题汇总'].fillna('', inplace = True)
	data['问题汇总_jieba'] = data['问题汇总'].apply(seg_sentence)
	data['解决方案汇总'].fillna('', inplace = True)
	data['解决方案汇总_jieba'] = data['解决方案汇总'].apply(seg_sentence)
	data['jieba'] = data['问题汇总_jieba'] + data['解决方案汇总_jieba']
	data = data.reset_index(drop = True)
	
	# 该仪器最近一年的维修次数
	def service_count_year(data):
	    instrument_id = data.iloc[-1]['设备编号']
	    cache_ins = data.loc[data['设备编号'] == instrument_id,:]    
	    cache_ins['year'] = cache_ins['问题发现日期'].apply(lambda x: x.year)
	    last_year = cache_ins.iloc[-1]['year']
	    return cache_ins['year'].value_counts()[last_year]
	service_count = []
	for ix in data.index:
	    data_time = data.iloc[:ix+1].copy()
	    service_count.append(service_count_year(data_time))
	data['最近一年维修次数'] = service_count

	# 该客户最近一年的维修总次数
	def client_service_count_year(data):
	    client_id = data.iloc[-1]['客户名称']
	    cache_ins = data.loc[data['客户名称'] == client_id,:]    
	    cache_ins['year'] = cache_ins['问题发现日期'].apply(lambda x: x.year)
	    last_year = cache_ins.iloc[-1]['year']
	    return cache_ins['year'].value_counts()[last_year]
	client_service_count = []
	for ix in data.index:
	    data_time = data.iloc[:ix+1].copy()
	    client_service_count.append(client_service_count_year(data_time))
	data['最近一年该客户的维修总次数'] = client_service_count

	# 该仪器最近一年的维修次数占对应客户最近一年的总维修次数的比
	data['年修占比'] = data['最近一年维修次数']/data['最近一年该客户的维修总次数']

	# 仪器距离上一次维修维护的时间间隔
	data.loc[pd.to_datetime(data['装机日期'])< pd.to_datetime('1999-05-01'), '装机日期'] = '2005-06-28'	
	data.loc[(data['装机日期'].isnull())&(data['装机年份'].notnull()),'装机日期'] = data.loc[(data['装机日期'].isnull())&(data['装机年份'].notnull()),'装机年份']
	data.loc[data['装机日期'] == '20XX','装机日期'] = '2016'
	data.loc[data['上次维修时间'].isnull(),'上次维修时间'] = data.loc[data['上次维修时间'].isnull(),'装机日期']
	data.loc[data['上次维修时间'] == '20XX','上次维修时间'] = '2016'
	
	# 该仪器的已使用年限
	days_cache = (pd.to_datetime(data['问题发现日期'], errors = 'coerce') - pd.to_datetime(data['装机日期'], errors = 'coerce')).apply(lambda x: x.days)
	data['已使用年限'] = round(days_cache/365.25,2)
	data.loc[data['已使用年限']<0, '已使用年限'] = 0

	# 同类仪器对应使用年限时的平均维修次数
	instrument_uniq = data.drop_duplicates(subset = '设备编号', keep = 'last')
	data_repair = data.loc[data['维修服务内容'] == '维修', :]
	data_repair['已使用年限_round'] = data_repair['已使用年限'].apply(lambda x: round(x))
	instrument_count = instrument_uniq['设备型号'].value_counts().to_dict()
	instrument_average_service = {}
	for instrument in instrument_count.keys():
	    instrument_average_service[instrument] = (data_repair.loc[data_repair['设备型号'] == instrument, '已使用年限_round'].value_counts()/instrument_count[instrument]).apply(lambda x: round(x, 3))
	average_service_stats = pd.concat(instrument_average_service,axis = 1)
	average_service_stats.loc['仪器台数'] = instrument_uniq['设备型号'].value_counts().to_dict()
	average_service_stats = average_service_stats.T.sort_values(by = '仪器台数', ascending = False).T
	average_service_stats.to_sql('instrument_average_service', engine, if_exists='replace',index= False)
	average_service_stats.drop('仪器台数', axis = 0, inplace = True)
	for col in average_service_stats.columns:
	    average_service_stats[col] = average_service_stats[col].fillna(average_service_stats[col].median())
	average_service_stats.fillna(0, inplace = True)
	average_service_map = average_service_stats.to_dict()
	average_service_map2 = {}
	for k1 in average_service_map.keys():
	    for k2 in average_service_map[k1].keys():
	        average_service_map2['{}_{}'.format(k1, k2)] = average_service_map[k1][k2]
	data['设备型号_已使用年限'] = data['设备型号'] + '_' + data['已使用年限_round'].astype(int).astype(str)
	data['设备年平均维修次数'] = pd.to_numeric(data['设备型号_已使用年限'].map(average_service_map2), errors = 'coerce')	
	data.drop(['设备型号_已使用年限', '已使用年限_round'], axis = 1, inplace = True)
	data.to_sql('fe_data1', engine, if_exists='replace',index= False)

	# # 紧急指数
	# X_jinji = data.loc[(data['维修服务内容'] == '维修')&(data['紧急程度'].isin(['正常','紧急'])),'jieba'].reset_index(drop = True)
	# y_jinji = data.loc[(data['维修服务内容'] == '维修')&(data['紧急程度'].isin(['正常','紧急'])),'紧急程度'].reset_index(drop = True).map({'正常': 0, '紧急': 1})	
	# lgb_jinji = lgb_w2v_train(X_jinji, y_jinji)
	# data['紧急指数'] = lgb_jinji.predict()
	# 
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=str, default= 'update')
    args = parser.parse_args()
    
    config = Config()
    feature_engineer()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from wordcloud import WordCloud
import jieba
from jieba import analyse
import re
from tqdm import tqdm
from sqlalchemy import create_engine
from config import Config
import os

jieba.load_userdict('../source/words/problem.txt')

def text_clean(content):
    try:
#         result = re.sub(r'[^\u4e00-\u9fa5,A-Za-z0-9]', " ",content)
        result = re.sub('&nbsp;', "",content)
        result = re.sub('<br>', "",result)
        result = re.sub(' ', "",result)
#         result = re.sub('正常', " ",result)
        result = re.sub('\n', "",result)
        result = re.sub('\t', "",result)
    except:
        result = ''
    return result

def word_cloud_tfidf(data_line, output, save_path):
    data_line_clean = data_line.apply(lambda x: text_clean(x))
    cache = ""
    for i in data_line_clean:
        cache = cache + i
    content = " ".join(jieba.cut(cache))
    key_words = analyse.extract_tags(content, topK=2000, withWeight=True, allowPOS=('nb','n','nr', 'ns','a','ad','an','nt','nz','v','d'))
    textrank = analyse.textrank(content,topK = 200,withWeight = True)
    keywords = dict()
    for i in key_words:
        keywords[i[0]] = i[1]
#     back_coloring = plt.imread(bg_image_path)  # 设置背景图片
    my_wordcloud = WordCloud(
                     background_color = 'white', #背景颜色
                     width=1500,height=960, #图片大小
                     margin=10
                   )
    my_wordcloud.generate_from_frequencies(keywords)
    plt.figure(figsize = (12, 7))
    plt.imshow(my_wordcloud)
    plt.axis("off")
    plt.savefig('{}/{}'.format(save_path, output))

def word_cloud(data_line):
    text = " "
    for i in data_line:
        text = text + i
    wordlist_after_jieba = jieba.cut(text, cut_all = True)
    wl_space_split = " ".join(wordlist_after_jieba)
    my_wordcloud = WordCloud().generate(wl_space_split)
    plt.figure(figsize = (12, 7))
    plt.imshow(my_wordcloud)
    plt.axis("off")
    plt.show()

def words2vec(words1=None, words2=None):
    """
    通过分词、计算并集、词频，得出词频向量
    :param words1:
    :param words2:
    :return: 词频的向量
    """
    v1 = []
    v2 = []
    tag1 = analyse.extract_tags(words1, withWeight=True)
    tag2 = analyse.extract_tags(words2, withWeight=True)
    tag_dict1 = {i[0]: i[1] for i in tag1}
    tag_dict2 = {i[0]: i[1] for i in tag2}
    merged_tag = set(tag_dict1.keys()) | set(tag_dict2.keys())
    for i in merged_tag:
        if i in tag_dict1:
            v1.append(tag_dict1[i])
        else:
            v1.append(0)
        if i in tag_dict2:
            v2.append(tag_dict2[i])
        else:
            v2.append(0)
    return v1, v2

def cosine_similarity(vector1, vector2):
    """
    余弦值相似度计算
    :param vector1:
    :param vector2:
    :return:
    """

    dot_product = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        norm1 += a ** 2
        norm2 += b ** 2
    if norm1 == 0.0 or norm2 == 0.0:
        return 0
    else:
        return round(dot_product / ((norm1 ** 0.5) * (norm2 ** 0.5)) * 100, 2)

def cosine_calculator(_data, col1, col2):
    cosine_sim = []
    for i in range(_data.shape[0]):
        w2v1, w2v2 = words2vec(_data[col1][i],_data[col2][i])
        cosine_sim.append(cosine_similarity(w2v1,w2v2))
    return cosine_sim 

def text_col_len(series):
    return series.apply(lambda x: len(x))

def text_concat(data, cut_value, col1, col2):
    _data = data.copy()
    cosine_sim = cosine_calculator(_data, col1, col2)
    _data['cosine_sim'] = cosine_sim
    _data['{}_len'.format(col1)] = text_col_len(_data[col1])
    _data['{}_len'.format(col2)] = text_col_len(_data[col2])
    _data['output'] = None
    _data.loc[_data['cosine_sim'] <=cut_value, 'output'] = _data.loc[_data['cosine_sim'] <=cut_value, col1] + _data.loc[_data['cosine_sim'] <cut_value,col2]
    _data.loc[(_data['cosine_sim'] >cut_value)&(_data['{}_len'.format(col1)] >= _data['{}_len'.format(col2)]), 'output'] = _data.loc[(_data['cosine_sim'] >cut_value)&(_data['{}_len'.format(col1)] >= _data['{}_len'.format(col2)]),col1]
    _data.loc[(_data['cosine_sim'] >cut_value)&(_data['{}_len'.format(col1)] < _data['{}_len'.format(col2)]), 'output'] = _data.loc[(_data['cosine_sim'] >cut_value)&(_data['{}_len'.format(col1)] < _data['{}_len'.format(col2)]),col2]
    return _data['output']

def data_load():
    data = {}
    for file in ['2015-2018维修数据_部分.xlsx', '2017维修数据.xlsx', '2018维修数据.xlsx']:
        data[file] = pd.read_excel('{}{}'.format(config.origin_data_filepath, file))
    data_ = pd.concat(data).reset_index(drop = True)[data['2015-2018维修数据_部分.xlsx'].columns]
    return data_

def data_preprocess(data_):
    data_.loc[-data_['紧急程度'].isin(['正常', '紧急', '重要', '无要求']), '紧急程度'] = '未知'
    service_content = {'维修':'维修', '仪器故障处理':'维修', '更换配件':'维修', 
                   '仪器维护保养':'维护保养', '维护':'维护保养', '年度维护':'维护保养',
                   '项目例行巡检':'巡检','巡检':'巡检',
                   '设备性能验证':'性能验证','性能验证':'性能验证',
                   '移机':'装机/移机', '搬迁':'装机/移机','搬迁调试':'装机/移机','装机调试':'装机/移机','装机调试':'装机/移机',
                   '软件问题处理':'其他','数据分析':'其他','试剂盒性能验证':'其他'}
    data_['维修服务内容'] = data_['维修服务内容'].replace().map(service_content)
    data_['维修服务内容'].fillna('待确认', inplace = True)

    data_equip = data_.drop_duplicates(subset = '设备编号', keep = 'last')
    zjrq_dict = data_equip['装机日期']
    zjrq_dict.index = data_equip['设备编号']
    zjrq_dict = zjrq_dict.to_dict()

    data = data_.loc[data_['设备编号'].notnull(),:].reset_index(drop = True)

    data['装机日期'] = data['设备编号'].map(zjrq_dict)

    data.loc[5162,'服务工时(小时)'] = 1
    data.loc[6123,'服务工时(小时)'] = 3
    data.loc[6124,'服务工时(小时)'] = 0.5
    data.loc[6427,'服务工时(小时)'] = 4
    data.loc[6721,'服务工时(小时)'] = 3
    data.loc[6842,'服务工时(小时)'] = 5.5

    data['服务间隔天数'] = (pd.to_datetime(data['上门日期'], errors = 'coerce') - pd.to_datetime(data['问题发现日期'], errors = 'coerce')).apply(lambda x: x.days)

    data['上次维修时间'] = data['问题发现日期'].copy()
    def last_service(x):
        del x[-1]
        x.insert(0, None)
        return x
    for equipment_id in data['设备编号'].unique():
        cache = list(data.loc[data['设备编号'] == equipment_id, '问题发现日期'])
        if len(cache) <= 1:
            data.loc[data['设备编号'] == equipment_id, '上次维修时间'] = None
        else:
            data.loc[data['设备编号'] == equipment_id, '上次维修时间'] = last_service(cache)

    data['装机日期'] = pd.to_datetime(data['装机日期'], errors = 'coerce')
    data.loc[(data['上次维修时间'].isnull())&(data['装机日期'] > '2016-06-30'),'上次维修时间'] = data.loc[(data['上次维修时间'].isnull())&(data['装机日期'] > '2016-06-30'),'装机日期']

    for col in ['问题描述', '故障描述','服务报告','解决方案','实际问题描述']:
        data[col] = data[col].apply(text_clean)    

    data['问题汇总1'] = text_concat(data, 85, '问题描述', '故障描述')
    data['问题汇总'] = text_concat(data, 85, '问题汇总1', '实际问题描述')
    data['解决方案汇总'] = text_concat(data, 85, '服务报告','解决方案')

    def find_year(x):
        if '20' in x:
            output = x[x.index('20'):x.index('20')+4]
        elif '19' in x:
            output = x[x.index('19'):x.index('19')+4]
        else:
            output = 'unknown'
        return output

    data['装机年份'] = data['设备编号'].apply(find_year)
    data = data.loc[data['装机年份'] != 'unknown',:].reset_index(drop = True)
    
    data.loc[data['设备编号'] == 'BS-2016-WTS-TQS-01','设备编号'] = 'BS-2016-WTS-TQD-01'
    data.loc[data['设备编号'] == 'BS-2016-WTS-TQD-01','设备型号'] = 'TQD'
    data.loc[data['设备编号'] == 'BS-2017-NS550-01', '设备型号'] = 'Nextseq 550AR'
    
    engine = create_engine('mysql+pymysql://%s:%s@%s:3306/%s?charset=utf8' %(config.sql_user, config.sql_password, config.sql_ip, config.sql_database),echo = False)
    data[['服务单号', '客户名称', '客户类型', '紧急程度', '问题发现日期', '分配工程师', 
      '设备编号', '设备型号','服务工时(小时)', '服务间隔天数', '上次维修时间', 
      '装机日期','装机年份', '维修服务内容', '问题汇总', '解决方案汇总']].to_sql('etl_data', engine, if_exists='replace',index= False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=str, default= 'update')
    args = parser.parse_args()
    
    config = Config()
    data_ = data_load()
    data_preprocess(data_)

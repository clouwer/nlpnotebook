
# coding: utf-8

import pandas as pd
from sqlalchemy import create_engine
import json
import time
import logging
import random

class _Sql_Base:
    
    # 加载sql数据库engine
    def create_engine(self, sql_dict):
        return create_engine('mysql+pymysql://%s:%s@%s:3306/%s?charset=utf8' %(sql_dict['user'], sql_dict['password'], sql_dict['ip'], sql_dict['database']),echo = False)

    # sql读表
    def table_load(self, table):
        sql = "select * from %s" % table
        return pd.read_sql_query(sql, self.engine)

    # sql表保存
    def table_save(self, table, table_name):
        table.to_sql(table_name, self.sql_engine, if_exists='replace',index= False)
    
    def json_load(self, file_path):
        with open(file_path) as json_file:
            return json.load(json_file)

    def json_write(self, file, file_path):
        with open(file_path, 'w') as json_file:
            json_file.write(json.dumps(file))

def get_logger(name):
    """logger
    """
    default_logger = logging.getLogger(name)
    default_logger.setLevel(logging.DEBUG)
    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")
    stream.setFormatter(formatter)
    default_logger.addHandler(stream)
    return default_logger


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
        table.to_sql(table_name, self.engine, if_exists='replace',index= False)
    
    def json_load(self, file_path):
        with open(file_path) as json_file:
            return json.load(json_file)

    def json_write(self, file, file_path):
        with open(file_path, 'w') as json_file:
            json_file.write(json.dumps(file))

class ProxyManager(object):
    """代理管理器
    """

    def __init__(self, proxies_or_path, interval_per_ip=0, is_single=False):
        '''
        @proxies_or_path, basestring or list, 代理path或列表
        @interval_per_ip, int, 每个ip调用最小间隔
        @is_single, bool, 是否启用单点代理,例如使用squid
        '''
        self.proxies_or_path = proxies_or_path
        self.host_time_map = {}
        self.interval = interval_per_ip
        self.is_single = is_single
        self.init_proxies(self.proxies_or_path)

    def init_proxies(self, proxies_or_path):
        '''初始化代理列表

        @proxies_or_path, list or basestring
        '''
        if self.is_single:
            self.proxies = proxies_or_path
        else:
            with open(proxies_or_path) as f:
                self.proxies = f.readlines()

    def reload_proxies(self):
        '''重新加载代理，proxies_or_path必须是文件路径
        '''
        if self.is_single:
            raise TypeError("is_single must be False!")
        with open(self.proxies_or_path) as f:
            self.proxies = f.readlines()
        logging.info("reload %s proxies ...", len(self.proxies))

    def get_proxy(self):
        '''获取一个可用代理

        如果代理使用过于频繁会阻塞，以防止服务器屏蔽
        '''
        # 如果使用单点代理,直接返回
        if self.is_single:
            return self.proxies
        proxy = self.proxies[random.randint(0, len(self.proxies) - 1)].strip()
        host, _ = proxy.split(':')
        latest_time = self.host_time_map.get(host, 0)
        interval = time.time() - latest_time
        if interval < self.interval:
            logging.info("%s waiting", proxy)
            time.sleep(self.interval)
        self.host_time_map[host] = time.time()
        return "http://%s" % proxy.strip()
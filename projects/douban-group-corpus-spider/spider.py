# coding: utf-8

import pandas as pd
from bs4 import BeautifulSoup
from config import GROUP_DICT, MAX_PAGE, SQL_DICT, HEADERS, PROXY_POOL_URL, MAX_GET_RETRY, OUTPUT_PATH, SPIDER_INTERVAL
from base import _Sql_Base
import requests
import emoji
import time
import random
import logging
import os
import argparse

class HTTPError(Exception):

    """ HTTP状态码不是200异常 """

    def __init__(self, status_code, url):
        self.status_code = status_code
        self.url = url

    def __str__(self):
        return "%s HTTP %s" % (self.url, self.status_code)

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

class Douban_corpus_spider(_Sql_Base):

    def __init__(self, is_proxy = False):

        self.GROUP_DICT = GROUP_DICT
        self.MAX_PAGE = MAX_PAGE
        self.sql_engine = self.create_engine(SQL_DICT)
        self.is_proxy = is_proxy
        if is_proxy:
            self.proxyIP = self.get_proxy()
        self.logger = get_logger("douban_spider")
            
    def request_douban(self, url):
        headers = {
            'User-Agent': HEADERS['GalaxyS5']
        }
        for i in range(MAX_GET_RETRY):
            try:
                if self.is_proxy:
                    proxyIP = self.proxyIP
                    proxies = {
                        'http' : proxyIP,
                        'https': proxyIP
                    }
                    response = requests.get(url, proxies=proxies, headers=headers)
                else:
                    response = requests.get(url, headers=headers)
                if response.status_code != 200:
                    raise HTTPError(response.status_code, url)
                print('successfully get data from %s' %url)
                break
            except Exception as exc:
                self.logger.warn("%s %d failed!\n%s", url, i, str(exc))
                if self.is_proxy:
                    self.proxyIP = self.get_proxy()
                continue
#         time.sleep(2)
        return response.text
    
    # 从代理池中随机取出一个IP
    def get_proxy(self):
        try:
            response = requests.get(PROXY_POOL_URL)
            if response.status_code == 200:
                print('proxy: %s' %response.text)
#                 time.sleep(1)
                return "http://%s" %response.text
        except ConnectionError:
            return None

    def spider_links(self, group, page):
        url = '{}discussion?start={}'.format(self.GROUP_DICT[group], str(page*25))
        html = self.request_douban(url)
        soup = BeautifulSoup(html, 'lxml')
        list_ = soup.find(class_='olt').find_all('tr')
        page_link = []; page_title = []
        for item in list_:
            try:
                page_link.append(item.find('a').get('href'))
                page_title.append(item.find('a').get('title'))
            except:
                continue
        return page_link, page_title

    def spider_page(self, url):
        html = self.request_douban(url)
        soup = BeautifulSoup(html, 'lxml')
        page_author_diag = []
        for item in soup.find(class_="note-content paper").find_all('p'):
            if len(item) > 0:
                page_author_diag.append(item.contents[0])
        list_ = soup.find_all(class_ = 'content')
        page_comments = []
        for item in list_:
            page_comments.append(item.contents[2].replace(' ','').replace('\n',''))
        return page_author_diag, page_comments

    def spider_group(self, group, max_page):
        spider_outputs = {}
        link_list = []
        title_list = []
        for page in range(max_page):
            link_list_page, title_list_page = self.spider_links(group, page)            
            link_list = link_list + link_list_page
            title_list = title_list + title_list_page
        for link in link_list:
            try:
                spider_outputs[link] = {}
                spider_outputs[link]['title'] = title_list[link_list.index(link)]
                spider_outputs[link]['author_diag'], spider_outputs[link]['comments'] = self.spider_page(link)
                self.json_write(spider_outputs, os.path.join(OUTPUT_PATH, '{}.json'.format(group)))
            except:
                print('%s fail' %link)
                continue        
        return spider_outputs

    def group_dict_transfer(self, output_dict):
        data = pd.DataFrame(output_dict).T
        data['link'] = data.index
        data = data.reset_index(drop = True)[['link','title','author_diag','comments']]
        data['author_diag'] = data['author_diag'].apply(lambda x: x[0] if x else '')
        data['comments'] = data['comments'].apply(lambda x: '|'.join(x))
        for col in ['title','author_diag','comments']:
            data[col] = data[col].apply(emoji.demojize)
        return data

    def run(self):
        for group in self.GROUP_DICT.keys():
            output_dict = self.spider_group(group, self.MAX_PAGE)
            output_table = self.group_dict_transfer(output_dict)
            self.table_save(output_table, group)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--proxy', type= int, default= 0)
    args = parser.parse_args()
    dcs = Douban_corpus_spider(is_proxy = args.proxy)
    dcs.run()
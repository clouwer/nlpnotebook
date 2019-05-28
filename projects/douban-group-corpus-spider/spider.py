
# coding: utf-8

import pandas as pd
from bs4 import BeautifulSoup
from config import GROUP_DICT, MAX_PAGE, SQL_DICT, HEADERS
from base import _Sql_Base, ProxyManager
import requests

class Douban_corpus_spider(_Sql_Base):

    def __init__(self):

        self.GROUP_DICT = GROUP_DICT
        self.MAX_PAGE = MAX_PAGE
        self.sql_engine = self.create_engine(SQL_DICT)
        self.ProxyManager = ProxyManager("./proxy_list.txt", 30)

    def request_douban(self, url):
        kwargs = {
            "headers": {
                "User-Agent": HEADERS,
                "Referer": "http://www.douban.com/"
            },
        }
        try:
            kwargs["proxies"] = {
                        "http": self.ProxyManager.get_proxy()}
            response = requests.get(url, **kwargs)
            if response.status_code == 200:
                return response.text
        except requests.RequestException:
            return None

    def spider_links(self, group_link, page):
        url = '{}discussion?start={}'.format(group_link, str(page*25))
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
        for item in soup.find(type="application/ld+json"):
            try:
                page_author_diag = json.loads(item)['text']
            except:
                page_author_diag = ''
        list_ = soup.find_all(class_='clearfix comment-item reply-item')
        page_comments = []
        for item in list_:
            try:
                page_comments.append(item.find('p').contents[0])
            except:
                continue
        return page_author_diag, page_comments

    def spider_group(self, group):
    	spider_outputs = {}
    	link_list = []
    	title_list = []
    	for page in range(self.MAX_PAGE):
    		link_list_page, title_list_page = self.spider_links(group, page)
    		link_list = link_list + link_list_page
    		title_list = title_list + title_list_page
    	for link in link_list:
    		spider_outputs[link] = {}
    		spider_outputs[link]['title'] = title_list[link_list.index(link)]
    		spider_outputs[link]['author_diag'], spider_outputs[link]['comments'] = self.spider_page(link)	
    	return spider_outputs

    def run(self):
        for group in self.GROUP_DICT.values():
            output_dict = self.spider_group(group)
            self.table_save(pd.DataFrame(output_dict).T, group)

def main():
    dcs = Douban_corpus_spider()
    dcs.run()

if __name__ == "__main__":
    main()


# coding: utf-8

import requests
from bs4 import BeautifulSoup
from config import GROUP_LIST, MAX_PAGE, OUTPUT_PATH
import json

def request_douban(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
    except requests.RequestException:
        return None

def spider_links(group_link, page):
    url = '{}discussion?start={}'.format(group_link, str(page*25))
    html = request_douban(url)
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

def spider_page(url):
    # page_link = 'https://www.douban.com/group/topic/138987835/'
    html = request_douban(url)
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

def spider_group(group):
	spider_outputs = {}
	link_list = []
	title_list = []
	for page in range(MAX_PAGE):
		link_list_page, title_list_page = spider_links(group, page)
		link_list = link_list + link_list_page
		title_list = title_list + title_list_page
	for link in link_list:
		spider_outputs[link] = {}
		spider_outputs[link]['title'] = title_list[link_list.index(link)]
		spider_outputs[link]['author_diag'], spider_outputs[link]['comments'] = spider_page(link)	
	return spider_outputs

def json_write(file, file_path):
    with open(file_path, 'w') as json_file:
        json_file.write(json.dumps(file))

def main():
	spider_outputs = {}
	for group in GROUP_LIST:
		spider_outputs[group] = spider_group(group)
	json_write(spider_outputs, OUTPUT_PATH)


if __name__ == "__main__":
    main()

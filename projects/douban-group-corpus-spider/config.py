
# coding: utf8

"""
配置
"""

# 响应头
HEADERS = {
'PC' : "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36",
'GalaxyS5' : "Mozilla/5.0 (iPhone; CPU iPhone OS 10_3_1 like Mac OS X) AppleWebKit/603.1.30 (KHTML, like Gecko) Version/10.0 Mobile/14E304 Safari/602.1"
}

# 豆瓣小组URL
GROUP_DICT = {
	'相互表扬小组': "https://www.douban.com/group/kuakua/"
	# '互相表扬': "https://www.douban.com/group/haogaoxiao/"
}

# 抓取前多少页
MAX_PAGE = 500

# 爬虫时间间隔
SPIDER_INTERVAL = 5

# 输出保存路径
OUTPUT_PATH = './static/'

# 数据库配置
SQL_DICT = {
	'user' : 'root',
	'password' : 'biosan#17',
	'ip' : '172.16.0.34',
	'database' : 'corpus_spider'
}

# 代理池地址
# PROXY_POOL_URL = 'http://localhost:5555/random'
PROXY_POOL_URL = 'http://http.tiqu.alicdns.com/getip3?num=1&type=3&pro=0&city=0&yys=100017&port=1&pack=54305&ts=0&ys=0&cs=0&lb=1&sb=0&pb=4&mr=1&regions=&gm=4'

# 代理访问重试次数
MAX_GET_RETRY = 20

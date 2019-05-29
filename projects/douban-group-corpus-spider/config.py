
# coding: utf8

"""
配置
"""

# 响应头
HEADERS = "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36"

# 豆瓣小组URL
GROUP_DICT = {
	'互相表扬夸夸小组': "https://www.douban.com/group/593625/",
	'相互表扬小组': "https://www.douban.com/group/kuakua/"
}

# 抓取前多少页
MAX_PAGE = 1

# 输出保存路径
OUTPUT_PATH = './static/'

# 数据库配置
SQL_DICT = {
	'user' : 'root',
	'password' : 'mysql123',
	'ip' : '172.16.0.164',
	'database' : 'corpus_spider'
}

# 代理池地址
PROXY_POOL_URL = 'http://localhost:5555/random'

# 代理访问重试次数
MAX_GET_RETRY = 20

# 好用代理IP

PROXY_LIST = ['177.54.97.249:3128']

# 响应头
HEADERS = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.67 Safari/537.36'


# 豆瓣小组URL
GROUP_DICT = {
	'互相表扬夸夸小组': "https://www.douban.com/group/593625/"
}

# 抓取前多少页
MAX_PAGE = 5

# 输出保存路径
OUTPUT_PATH = 'C:\\Jupyter\\nlpnotebook\\projects\\spider-douban-group\\static\\output.json'

# 数据库配置
SQL_DICT = {
	'user' : 'root',
	'password' : 'mysql123',
	'ip' : '172.16.0.164',
	'database' : 'corpus_spider'
}

# 代理池地址
PROXY_POOL_URL = 'http://localhost:5555/random'
## jieba知识整理



[github原网址](https://github.com/fxsjy/jieba/tree/master/test)



### 使用算法

结巴中文分词涉及到的算法包括：

1. 基于Trie树结构实现高效的词图扫描，生成句子中汉字所有可能成词情况所构成的有向无环图（DAG)；
2. 采用了动态规划查找最大概率路径, 找出基于词频的最大切分组合；
3. 对于未登录词，采用了基于汉字成词能力的HMM模型，使用了Viterbi算法。



### 常用功能实现

#### 分词

结巴中文分词支持的三种分词模式包括：

1. 精确模式：试图将句子最精确地切开，适合文本分析；cut_all=False

2. 全模式：把句子中所有的可以成词的词语都扫描出来, 速度非常快，但是不能解决歧义问题；cut_all=True

3. 搜索引擎模式：在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词。

   同时结巴分词支持繁体分词和自定义字典方法。  jieba.cut_for_search()

```python
import jieba

seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
# join是split的逆操作
# 即使用一个拼接符将一个列表拼成字符串
print("/ ".join(seg_list))  # 全模式

seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
print("/ ".join(seg_list))  # 精确模式

seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
print("/ ".join(seg_list))

seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
print("/ ".join(seg_list))
```



结巴有新词识别能力，但自行添加新词可以保证更高的正确率，尤其是专有名词。

用法： 

```python
jieba.load_userdict(file_name)  # file_name 为文件类对象或自定义词典的路径
```

词典格式和 dict.txt 一样，一个词占一行；

每一行分三部分：词语、词频（可省略）、词性（可省略），用空格隔开，顺序不可颠倒。

file_name 若为路径或二进制方式打开的文件，则文件必须为 UTF-8 编码。

词频省略时使用自动计算的能保证分出该词的词频。



file_name可参考格式：

```txt
云计算 5
李小福 2 nr
创新办 3 i
easy_install 3 eng
好用 300
韩玉赏鉴 3 nz
八一双鹿 3 nz
台中
凱特琳 nz
Edu Trust认证 2000
```



#### 调整词典

使用 add_word(word, freq=None, tag=None) 和 del_word(word) 可在程序中动态修改词典。
使用 suggest_freq(segment, tune=True) 可调节单个词语的词频，使其能（或不能）被分出来。

注意：自动计算的词频在使用 HMM 新词发现功能时可能无效。

```python
print('/'.join(jieba.cut('如果放到post中将出错。', HMM=False)))
如果/放到/post/中将/出错/。
jieba.suggest_freq(('中', '将'), True)
494
print('/'.join(jieba.cut('如果放到post中将出错。', HMM=False)))
如果/放到/post/中/将/出错/。
print('/'.join(jieba.cut('「台中」正确应该不会被切开', HMM=False)))
「/台/中/」/正确/应该/不会/被/切开
jieba.suggest_freq('台中', True)
69
print('/'.join(jieba.cut('「台中」正确应该不会被切开', HMM=False)))
「/台中/」/正确/应该/不会/被/切开
```

“通过用户自定义词典来增强歧义纠错能力” — https://github.com/fxsjy/jieba/issues/14



#### 去除停用词

主要思想是分词过后，遍历一下停用词表，去掉停用词。

```python
import jieba  
  
# jieba.load_userdict('userdict.txt')  
# 创建停用词list  
def stopwordslist(filepath):  
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]  
    return stopwords  
  
  
# 对句子进行分词  
def seg_sentence(sentence):  
    sentence_seged = jieba.cut(sentence.strip())  
    stopwords = stopwordslist('./test/stopwords.txt')  # 这里加载停用词的路径  
    outstr = ''  
    for word in sentence_seged:  
        if word not in stopwords:  
            if word != '\t':  
                outstr += word  
                outstr += " "  
    return outstr
```

可参考的停用词表

中文常用停用词表（哈工大停用词表、百度停用词表等）：https://github.com/goto456/stopwords



#### Tokenize：返回词语在原文的起始位置

```python
result = jieba.tokenize(u'永和服装饰品有限公司')
for tk in result:
    print "word %s\t\t start: %d \t\t end:%d" % (tk[0],tk[1],tk[2])
word 永和                start: 0                end:2
word 服装                start: 2                end:4
word 饰品                start: 4                end:6
word 有限公司            start: 6                end:10
```



#### 词性标注

词性指以词的特点作为划分词类的根据。现代汉语的词可以分为两类14种词性。

```python
# 加载jieba.posseg并取个别名，方便调用

import jieba.posseg as pseg
words = pseg.cut("我爱北京天安门")
for word, flag in words:
    # 格式化模版并传入参数
    print('%s, %s' % (word, flag))
```

对应词性表见[使用Jieba进行中文词性标注](https://blog.csdn.net/u013421629/article/details/82428539)



### 参考资料

[Python第三方库jieba（中文分词）入门与进阶（官方文档）](https://blog.csdn.net/qq_34337272/article/details/79554772)

[python结巴分词、jieba加载停用词表](https://blog.csdn.net/u012052268/article/details/77825981)

[Python 中文 文本分析 实战：jieba分词+自定义词典补充+停用词词库补充+词频统计](https://blog.csdn.net/qq_30262201/article/details/80128076)

[使用Jieba工具中文分词及文本聚类概念](http://www.cnblogs.com/eastmount/p/5055906.html)

视频教程：[使用jieba分词处理文本](https://study.163.com/course/courseLearn.htm?courseId=1003520028#/learn/video?lessonId=1004015740&courseId=1003520028)

[使用Jieba进行中文词性标注](https://blog.csdn.net/u013421629/article/details/82428539)
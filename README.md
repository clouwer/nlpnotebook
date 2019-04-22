# nlp

我的NLP学习进展

如何设计出一个更work的task-specific的网络？



### 未消化完的学习资料

[Data Science Challenge / Competition Deadlines](https://github.com/iphysresearch/DataSciComp)

[Kaggle Past Solutions](http://ndres.me/kaggle-past-solutions/)

[Data competition Top Solution 数据竞赛Top解决方案开源整理](https://github.com/Smilexuhc/Data-Competition-TopSolution)

[文本关键词提取算法总结和Python实现](https://zhuanlan.zhihu.com/p/49049482)

[Python标准库01 正则表达式 (re包)](http://www.cnblogs.com/vamei/archive/2012/08/31/2661870.html)

[文本分析_机器学习PAI-阿里云](https://help.aliyun.com/document_detail/42747.html?spm=a2c4g.11186623.6.554.sGWW2U#%E5%85%B3%E9%94%AE%E8%AF%8D%E6%8A%BD%E5%8F%96)

[练手|常见30种NLP任务的练手项目](https://zhuanlan.zhihu.com/p/51279338)

[NLP能解决语义问题吗？](https://zhuanlan.zhihu.com/p/44023294)

[谈谈数据科学](https://zhuanlan.zhihu.com/p/38198345)

[中文分词是个伪问题](https://zhuanlan.zhihu.com/p/54499197)

[初入NLP领域的一些小建议](https://zhuanlan.zhihu.com/p/59184256)

[NLP自然语言处理从入门到迷茫](https://zhuanlan.zhihu.com/p/32951278)

[word2vec前世今生](https://www.cnblogs.com/iloveai/p/word2vec.html)

[学术派整理,一份从基础到实战的 NLP 学习清单~](https://zhuanlan.zhihu.com/p/58687602)

[自然语言处理是如何工作的？一步步教你构建 NLP 流水线
机器之心](https://zhuanlan.zhihu.com/p/41850756)

[蚂蚁金融NLP竞赛——文本语义相似度赛题总结](https://zhuanlan.zhihu.com/p/51675979)

[别求面经了！小夕手把手教你斩下NLP算法岗offer！（19.3.21更新）](https://zhuanlan.zhihu.com/p/45802662)

[如何解决90％的NLP问题：逐步指导](https://zhuanlan.zhihu.com/p/57658502)

[NLP 教程：词性标注、依存分析和命名实体识别解析与应用](https://ai.yanxishe.com/page/TextTranslation/848)

**[语音语义字幕组](https://ai.yanxishe.com/page/translateGroup/6)**

[词性标注](http://www.hankcs.com/nlp/part-of-speech-tagging.html)



[利用Python实现中文文本关键词抽取的三种方法](https://github.com/AimeeLee77/keyword_extraction)

[以 gensim 訓練中文詞向量](https://zake7749.github.io/2016/08/28/word2vec-with-gensim/)

[SnowNLP: Simplified Chinese Text Processing](https://github.com/isnowfy/snownlp)



### NLP任务总结

自然语言处理（简称NLP），是研究计算机处理人类语言的一门技术，包括：

1.**句法语义分析**：对于给定的句子，进行分词、词性标记、命名实体识别和链接、句法分析、语义角色识别和多义词消歧。

2.**信息抽取**：从给定文本中抽取重要的信息，比如，时间、地点、人物、事件、原因、结果、数字、日期、货币、专有名词等等。通俗说来，就是要了解谁在什么时候、什么原因、对谁、做了什么事、有什么结果。涉及到实体识别、时间抽取、因果关系抽取等关键技术。

3.**文本挖掘**（或者文本数据挖掘）：包括文本聚类、分类、信息抽取、摘要、情感分析以及对挖掘的信息和知识的可视化、交互式的表达界面。目前主流的技术都是基于统计机器学习的。

4.**机器翻译**：把输入的源语言文本通过自动翻译获得另外一种语言的文本。根据输入媒介不同，可以细分为文本翻译、语音翻译、手语翻译、图形翻译等。机器翻译从最早的基于规则的方法到二十年前的基于统计的方法，再到今天的基于神经网络（编码-解码）的方法，逐渐形成了一套比较严谨的方法体系。

5.**信息检索**：对大规模的文档进行索引。可简单对文档中的词汇，赋之以不同的权重来建立索引，也可利用1，2，3的技术来建立更加深层的索引。在查询的时候，对输入的查询表达式比如一个检索词或者一个句子进行分析，然后在索引里面查找匹配的候选文档，再根据一个排序机制把候选文档排序，最后输出排序得分最高的文档。

6.**问答系统**： 对一个自然语言表达的问题，由问答系统给出一个精准的答案。需要对自然语言查询语句进行某种程度的语义分析，包括实体链接、关系识别，形成逻辑表达式，然后到知识库中查找可能的候选答案并通过一个排序机制找出最佳的答案。

7.**对话系统**：系统通过一系列的对话，跟用户进行聊天、回答、完成某一项任务。涉及到用户意图理解、通用聊天引擎、问答引擎、对话管理等技术。此外，为了体现上下文相关，要具备多轮对话能力。同时，为了体现个性化，要开发用户画像以及基于用户画像的个性化回复。



- 简单的任务：拼写检查，关键词检索，同义词检索等
- 复杂一点的任务：信息提取（比如从网页中提取价格，产地，公司名等信息），情感分析，文本分类等
- 更复杂的任务：机器翻译，人机对话，QA系统



### 知识网络



##### 机器学习方法

LogisticRegression，LinearSVC，LightGbm……



##### 深度学习方法

Attention，AttentionRNN，Capsule，Convlstm，Dpcnn，Lstmgru，RCNN，SimpleCNN，SnapshotCallback，TextCNN，TextRNN……



### 环境搭建

Google Colab:   [《Google Colab配置记录》](md/Google Colab配置记录.md)

- [Google Colab网址](https://colab.research.google.com/notebooks/welcome.ipynb)
- [Google Colab使用Tips](https://blog.csdn.net/weixin_42441790/article/details/86748345)

深度学习服务器：[《服务器运维知识整理》](md/聊天机器人知识整理.md)

- [如何与深度学习服务器优雅的交互？（长期更新）](https://zhuanlan.zhihu.com/p/32496193)

Win10下的环境搭建：

[Win10下安装Anaconda+CUDA+cudnn+TensorFlow+keras+PyTorch+Pycharm](https://www.jianshu.com/p/9f89633bad57)



### Github库

[Information-Extraction-Chinese](https://github.com/crownpku/Information-Extraction-Chinese)： 中文实体识别与关系提取

[SnowNLP: Simplified Chinese Text Processing](https://github.com/isnowfy/snownlp)

NLP工具包大全 ：https://github.com/fighting41love/funNLP

ChineseNER：https://github.com/zjy-ucas/ChineseNER

算法/深度学习/NLP面试笔记：https://github.com/imhuay/Algorithm_Interview_Notes-Chinese

NLP-BERT 谷歌自然语言处理模型：BERT-基于pytorch：https://github.com/Y1ran/NLP-BERT--ChineseVersion

text_classification：https://github.com/brightmart/text_classification

cocoNLP ： https://github.com/fighting41love/cocoNLP

>  人名、地址、邮箱、手机号、手机归属地 等信息的抽取，rake短语抽取算法。

Chinese Word Vectors 中文词向量：https://github.com/Embedding/Chinese-Word-Vectors

基于医疗领域知识图谱的问答系统：https://github.com/zhihao-chen/QASystemOnMedicalGraph

https://github.com/zhihao-chen/QASystemOnMedicalKG

Eliyar.Blog：https://eliyar.biz/archives/

Kashgari：https://github.com/BrikerMan/Kashgari

bert-as-service ：https://github.com/hanxiao/bert-as-service 

CDCS 中国数据竞赛优胜解集锦：https://github.com/geekinglcq/CDCS

HanLP：https://github.com/hankcs/pyhanlp



### Blog

刘焕勇：https://liuhuanyong.github.io/



### 数据集

[自然语言处理（NLP）数据集整理](https://zhuanlan.zhihu.com/p/35423943)

[中文语料库1](https://github.com/brightmart/nlp_chinese_corpus)

[中文公开聊天语料库](https://github.com/codemayq/chaotbot_corpus_Chinese)

[OpenKG.CN: 开放的中文知识图谱](http://www.openkg.cn/)

[自然语言处理 怎么获得数据集 中文语料集？](https://blog.csdn.net/u012052268/article/details/78035272)

NER中文语料：https://github.com/yaleimeng/NER_corpus_chinese

CCKS2017电子病历实体标注：https://github.com/liuhuanyong/MedicalNamedEntityRecognition

100+ Chinese Word Vectors 上百种预训练中文词向量：https://github.com/Embedding/Chinese-Word-Vectors

天池瑞金知识图谱：https://github.com/ZhengZixiang/tianchi_ruijin_knowledge_graph



### 常用工具

[用re.sub做文本预处理](https://blog.csdn.net/johline/article/details/78802381)

jieba：[《jieba知识整理》](md/jieba知识整理.md)

word2vec: [《word2vec知识整理》](md/word2vec知识整理.md)

文本去重：[《文本去重方法知识整理》](md/文本去重方法知识整理.md)

pyhanlp：https://github.com/hankcs/pyhanlp

自然语言处理工具包HanLP的Python接口 <http://hanlp.hankcs.com/>

[NLP可视化: 用Python生成词云](https://zhuanlan.zhihu.com/p/23453890)

- [词云库wordcloud中文乱码解决办法](https://blog.csdn.net/Dick633/article/details/80261233)
- [简单TFIDF词云实现代码](md/常用工具实现代码/word_cloud.md)

[Python微信库:itchat的用法详解](http://www.php.cn/python-tutorials-394725.html)

- [图灵机器人官网](http://www.tuling123.com/member/robot/index.jhtml)
- [微信图灵机器人实现](md/常用工具实现代码/tuling.py)

Neo4j：知识图谱工具Py2Neo [Neo4j简介及Py2Neo的用法](https://cuiqingcai.com/4778.html)

BERT：

- 团队预训练好的[BERT-base Chinese](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)模型（364.20MB）
- [BERT](https://github.com/google-research/bert)
- [干货 | BERT fine-tune 终极实践教程](https://www.jianshu.com/p/aa2eff7ec5c1)
- [【NLP】BERT中文实战踩坑](https://zhuanlan.zhihu.com/p/51762599)
- bert论文中文翻译: [link](https://github.com/yuanxiaosc/BERT_Paper_Chinese_Translation)
- bert原作者的slides: [link](https://pan.baidu.com/s/1OSPsIu2oh1iJ-bcXoDZpJQ) 提取码: iarj
- 文本分类实践: [github](https://github.com/NLPScott/bert-Chinese-classification-task)
- bert tutorial文本分类教程: [github](https://github.com/Socialbird-AILab/BERT-Classification-Tutorial)
- bert pytorch实现: [github](https://github.com/huggingface/pytorch-pretrained-BERT)
- bert用于中文命名实体识别 tensorflow版本: [github](https://github.com/macanv/BERT-BiLSTM-CRF-NER) [说明文档](https://blog.csdn.net/macanv/article/details/85684284)
- 使用预训练语言模型BERT做中文NER：[github](https://github.com/ProHiryu/bert-chinese-ner)
- BERT生成句向量，BERT做文本分类、文本相似度计算[github](https://github.com/terrifyzhao/bert-utils)  [BERT完全指南](https://blog.csdn.net/u012526436/article/details/86296051)
- bert 基于 keras 的封装分类标注框架 Kashgari，几分钟即可搭建一个分类或者序列标注模型: [github](https://github.com/BrikerMan/Kashgari)
- bert、ELMO的图解： [github](https://jalammar.github.io/illustrated-bert/)
- BERT: Pre-trained models and downstream applications: [github](https://github.com/asyml/texar/tree/master/examples/bert)
- 使用BERT生成句向量：[link](https://blog.csdn.net/u012526436/article/details/87697242)
- bert-as-service:  [github](https://github.com/hanxiao/bert-as-service )    https://zhuanlan.zhihu.com/p/50582974 
- 用BERT进行序列标记和文本分类的模板代码：[github](https://github.com/yuanxiaosc/BERT-for-Sequence-Labeling-and-Text-Classification)



### 技术专题

文本表示：[《文本表示知识整理》](md/文本表示知识整理.md)



### 应用场景

中文聊天机器人：[《聊天机器人知识整理》](md/聊天机器人知识整理.md)

- [从产品完整性的角度浅谈chatbot](https://zhuanlan.zhihu.com/p/34927757)
- [基于Rasa_NLU的微信chatbot](http://rowl1ng.com/%E6%8A%80%E6%9C%AF/chatbot.html)
- [【教程】从零开始动手实现微信聊天机器人](https://www.bilibili.com/video/av16505671/)
- [基于中文的rasa_nlu](https://github.com/crownpku/rasa_nlu_chi)
- [深度学习对话系统实战篇--简单chatbot代码实现](https://zhuanlan.zhihu.com/p/32455898)
- [深度学习对话系统实战篇--新版本chatbot代码实现](https://zhuanlan.zhihu.com/p/32801792)
- [给chatbot融入人格特征--论文阅读](https://zhuanlan.zhihu.com/p/49447966)

智能问答算法

- [智能问答算法原理及实践之路 - 腾讯小和](md/智能问答算法原理及实践之路.pdf)



### 问题细分

不同细分问题间存在交集，比如部分意图识别和情感分析问题也可归为文本分类问题来解决



NER (Named Entity Recognition，命名实体识别):

- Stanford NLP

- [CRF：条件随机场](https://www.cnblogs.com/Determined22/p/6915730.html)

- [命名实体识别（NER）的二三事](https://www.sohu.com/a/148858736_500659)
- [达观数据：如何打造一个中文NER系统](http://zhuanlan.51cto.com/art/201705/540693.htm)
- [Google Colab实战-基于Google BERT的中文命名实体识别（NER）](https://blog.csdn.net/weixin_42441790/article/details/86751031)
- [BERT+BiLSTM-CRF-NER用于做ner识别](https://blog.csdn.net/luoyexuge/article/details/84728649)
- [达观数据：一文详解深度学习在命名实体识别(NER)中的应用](https://www.jiqizhixin.com/articles/2018-08-31-2)
- [神圣的NLP！一文理解词性标注、依存分析和命名实体识别任务](https://zhuanlan.zhihu.com/p/42721891)
- [NLP - 基于 BERT 的中文命名实体识别（NER)](https://eliyar.biz/nlp_chinese_bert_ner/)
- zh-NER-TF : [github](https://github.com/Determined22/zh-NER-TF)     [说明文档](https://www.cnblogs.com/Determined22/p/7238342.html)
- ChineseNER：[github](https://github.com/zjy-ucas/ChineseNER)
- ccks2017：[github](https://github.com/liuhuanyong/MedicalNamedEntityRecognition)



意图识别：[《意图识别知识整理》](md/意图识别知识整理.md)

- [NLP系列学习：意图识别](https://zhuanlan.zhihu.com/p/41944121)
- [基于fastText的意图识别框架](https://zhuanlan.zhihu.com/p/53297108)



文本分类：[《文本分类知识整理》](md/文本分类知识整理.md)

自然语言生成：
[Ehud Reiter教授的博客](https://ehudreiter.com/) 北大万小军教授强力推荐，该博客对NLG技术、评价与应用进行了深入的探讨与反思。
[文本生成相关资源大列表](https://github.com/ChenChengKuan/awesome-text-generation)
[自然语言生成：让机器掌握自动创作的本领 - 开放域对话生成及在微软小冰中的实践](https://drive.google.com/file/d/1Mdna3q986k6OoJNsfAHznTtnMAEVzv5z/view)
[文本生成控制](https://github.com/harvardnlp/Talk-Latent/blob/master/main.pdf)



### 比赛



#### 脱敏数据

[“达观杯”文本智能处理挑战赛](md/达观文本智能处理挑战.md)



#### 非脱敏数据

[第二届搜狐内容识别算法大赛](md/搜狐第二届算法大赛.md)

[2018 CCL-中移在线客服领域用户意图分类](https://github.com/nlpjoe/2018-CCL-UIIMCS)



### nlp个人项目页

#### 进行中的

[Biosan维修部维护效率提升项目](https://github.com/kenshinpg/nlpnotebook/tree/master/projects/Biosan-service-upgrade-2019)


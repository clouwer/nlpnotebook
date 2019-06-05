## Word2vec知识整理



https://radimrehurek.com/gensim/models/word2vec.html

### gensim.word2vec参数解释

class gensim.models.word2vec.Word2Vec(sentences=None,size=100,alpha=0.025,window=5, min_count=5, max_vocab_size=None, sample=0.001,seed=1, workers=3,min_alpha=0.0001, sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=<built-in function hash>,iter=5,null_word=0, trim_rule=None, sorted_vocab=1, batch_words=10000)



- `sentences`：训练的语料，一个可迭代对象。对于从磁盘加载的大型语料最好用gensim.models.word2vec.BrownCorpus，gensim.models.word2vec.Text8Corpus ，gensim.models.word2vec.LineSentence 去生成sentences
- `sg`： 用于设置训练算法，默认为0，对应CBOW算法；sg=1则采用skip-gram算法。
- `size`：是指特征向量的维度，神经网络 NN 层单元数，它也对应了训练算法的自由程度。默认为100。大的size需要更多的训练数据,但是效果会更好. 推荐值为几十到几百。
- `window`：表示当前词与预测词在一个句子中的最大距离是多少。Harris 在 1954 年提出的分布假说( distributional hypothesis)指出， 一个词的词义由其所在的上下文决定。所以word2vec的参数中，窗口设置一般是5，而且是左右随机1-5（小于窗口大小）的大小，是均匀分布,随机的原因应该是比固定窗口效果好，增加了随机性，个人理解应该是某一个中心词可能与前后多个词相关，也有的词在一句话中可能只与少量词相关（如短文本可能只与其紧邻词相关）。
- `alpha`： 在随机梯度下降法中迭代的初始步长。算法原理篇中标记为η，默认是0.025。
- `min_alpha`: 由于算法支持在迭代的过程中逐渐减小步长，min_alpha给出了最小的迭代步长值。随机梯度下降中每轮的迭代步长可以由iter，alpha， min_alpha一起得出。这部分由于不是word2vec算法的核心内容，因此在原理篇我们没有提到。对于大语料，需要对alpha, min_alpha,iter一起调参，来选择合适的三个值。
- `seed`：用于随机数发生器。与初始化词向量有关。
- `min_count`： 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5。该模块在训练结束后可以通过调用model.most_similar('电影',topn=10)得到与电影最相似的前10个词。如果‘电影’未被训练得到，则会报错‘训练的向量集合中没有留下该词汇’。需要计算词向量的最小词频。这个值可以去掉一些很生僻的低频词，默认是5。如果是小语料，可以调低这个值。
- `max_vocab_size`： 设置词向量构建期间的RAM限制。如果所有独立单词个数超过这个，则就消除掉其中最不频繁的一个。每一千万个单词需要大约1GB的RAM。设置成None则没有限制。
- `sample`： 高频词汇的随机降采样的配置阈值，默认为1e-3，范围是(0,1e-5)
- `workers`参数控制训练的并行数。该参数只有在机器已安装 Cython 情况下才会起到作用。如没有 Cython，则只能单核运行。
- `hs`： 即我们的word2vec两个解法的选择了，如果是0， 则是Negative Sampling，是1的话并且负采样个数negative大于0， 则是Hierarchical Softmax。默认是0即Negative Sampling。
- `negative`： 即使用Negative Sampling时负采样的个数，默认是5。推荐在[3,10]之间。这个参数在我们的算法原理篇中标记为neg。如果>0,则会采用negativesampling，用于设置多少个noise words
- `cbow_mean`： 仅用于CBOW在做投影的时候，为0，则算法中的xwxw为上下文的词向量之和，为1则为上下文的词向量的平均值。在我们的原理篇中，是按照词向量的平均值来描述的。个人比较喜欢用平均值来表示xw，默认值也是1,不推荐修改默认值。
- `hashfxn`： hash函数来初始化权重。默认使用python的hash函数
- `iter`：随机梯度下降法中迭代的最大次数，默认是5。对于大语料，可以增大这个值。
- `trim_rule`： 用于设置词汇表的整理规则，指定那些单词要留下，哪些要被删除。可以设置为None（min_count会被使用）或者一个接受()并返回RUlE_DISCARD,utils.RUlE_KEEP或者utils.RUlE_DEFAUlT的函数。
- `sorted_vocab`： 如果为1（default），则在分配word index 的时候会先对单词基于频率降序排序。
- `batch_words`：每一批的传递给线程的单词的数量，默认为10000



### 调参



1. 架构：skip-gram（慢、对罕见字有利）vs CBOW（快）

   可以看出，skip-gram进行预测的次数是要多于cbow的：因为每个词在作为中心词时，都要使用周围词进行预测一次。这样相当于比cbow的方法多进行了K次（假设K为窗口大小），因此时间的复杂度为O(KV)，训练时间要比cbow要长。

2. 训练算法：分层softmax（对罕见字有利）vs 负采样（对常见词和低纬向量有利）

3. 负例采样准确率提高，速度会慢，不使用negative sampling的word2vec本身非常快，但是准确性并不高

4. 欠采样频繁词：可以提高结果的准确性和速度（适用范围1e-3到1e-5）

5. 文本（window）大小：skip-gram通常在10附近，CBOW通常在5附近
   

   

### 应用细节

#### 输入

用 Python 内置的 list 类型作为输入很方便，但当输入内容较多时，会占用很大的内存空间。Gemsim 的输入只要求序列化的句子，而不需要将所有输入都存储在内存中。简单来说，可以输入一个句子，处理它，删除它，再载入另外一个句子。

举例来说， 假如输入分散在硬盘的多个文件中，每个句子一行，那么不需要将所有输入先行存储在内存中，Word2vec 可以一个文件一个文件，一行一行地进行处理。

```
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()
 
sentences = MySentences('/some/directory') # a memory-friendly iterator
model = gensim.models.Word2Vec(sentences)
```

如果希望对文件中的内容进行预处理，举例来说，转换编码，大小写转换，去除数字等操作，均可以在 `MySentences` 迭代器中完成，完全独立于 Word2vec。Word2vec 只负责接收 yield 的输入。

**针对高级用户**：调用 `Word2Vec(sentences, iter=1)` 会调用句子迭代器运行两次（一般来说，会运行 `iter+1` 次，默认情况下   `iter=5`）。第一次运行负责收集单词和它们的出现频率，从而构造一个内部字典树。第二次以及以后的运行负责训练神经模型。这两次运行（`iter+1`）也可以被手动初始化，如果输入流是无法重复利用的，也可以用下面的方式对其进行初始化。

```
model = gensim.models.Word2Vec(iter=1)  # an empty model, no training yet
model.build_vocab(some_sentences)  # can be a non-repeatable, 1-pass generator
model.train(other_sentences)  # can be a non-repeatable, 1-pass generator
```

如果对 Python 中迭代器，可迭代的，生成器这些概念不是很理解，可以参考下文。
 [Python关键字yield的解释](https://link.jianshu.com?t=http://pyzh.readthedocs.io/en/latest/the-python-yield-keyword-explained.html#id8)



#### 内存

在内部，Word2vec 模型的参数以矩阵形式存储（NumPy 数组），数组的大小为 *#vocabulary* 乘以 *#size* 的浮点数 （4 *bytes*）。

三个如上的矩阵被存储在内存中（将其简化为两个或一个的工作进行中）。如果输入中存在 100,000 个互异的词，神经网络规模 `size` 设为200，则该模型大致需要内存
 `100,000 * 200 * 3 * 4 bytes = ~229MB`。

除此之外，还需要一些额外的空间存储字典树，但除非输入内容极端长，内存主要仍被上文所提到的矩阵所占用。



#### 存储和载入模型

完成训练后只存储并使用`~gensim.models.keyedvectors.KeyedVectors`
该模型可以通过以下方式存储/加载：
`~gensim.models.word2vec.Word2Vec.save` 保存模型
`~gensim.models.word2vec.Word2Vec.load` 加载模型

训练过的单词向量也可以从与其兼容的格式存储/加载：
`gensim.models.keyedvectors.KeyedVectors.save_word2vec_format`实现原始 word2vec 的保存

`gensim.models.keyedvectors.KeyedVectors.load_word2vec_format` 单词向量的加载



#### 模型的属性

- `wv`： 是类 `~gensim.models.keyedvectors.Word2VecKeyedVectors`生产的对象，在word2vec是一个属性。
为了在不同的训练算法（Word2Vec，Fastext，WordRank，VarEmbed）之间共享单词向量查询代码，gensim将单词向量的存储和查询分离为一个单独的类 KeyedVectors
包含单词和对应向量的映射。可以通过它进行词向量的查询

```python
model_w2v.wv.most_similar("民生银行")  # 找最相似的词
model_w2v.wv.get_vector("民生银行")  # 查看向量
model_w2v.wv.syn0  #  model_w2v.wv.vectors 一样都是查看向量
model_w2v.wv.vocab  # 查看词和对应向量
model_w2v.wv.index2word  # 每个index对应的词
```

> 小提示：
> 需要注意的是word2vec采用的是标准hash table存放方式，hash码重复后挨着放 取的时候根据拿出index找到词表里真正单词，对比一下
> `syn0` ：就是词向量的大矩阵，第i行表示vocab中下标为i的词
> `syn1`：用hs算法时用到的辅助矩阵，即文章中的Wx
> `syn1neg`：negative sampling算法时用到的辅助矩阵
> `Next_random`：作者自己生成的随机数，线程里面初始化就是：
>
> 

`vocabulary`：是类 `~gensim.models.word2vec.Word2VecVocab`模型的词汇表,除了存储单词外，还提供额外的功能，如构建一个霍夫曼树（频繁的单词更接近根），或丢弃极其罕见的单词。

`trainables` 是类 `~gensim.models.word2vec.Word2VecTrainables`训练词向量的内部浅层神经网络，CBOW和skip-gram(SG)略有不同，它的weights就是我们后面需要使用的词向量，隐藏层的size和词向量特征size一致



#### sentences相关

训练首先是语料集的加载。首先要生成Word2Vec需要的语料格式：

1. 对于简单的句子可以：

```python
from gensim.models import Word2Vec
# sentences只需要是一个可迭代对象就可以
sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
model = Word2Vec(sentences, min_count=1)  # 执行这一句的时候就是在训练模型了
```

2. 对于大型语料库：

   Gemsim 的输入只要求序列化的句子，而不需要将所有输入都存储在内存中。简单来说，可以输入一个句子，处理它，删除它，再载入另外一个句子。
   `gensim.models.word2vec.BrownCorpus: BrownCorpus`是一个英国语料库，可以用这个直接处理
   `gensim.models.word2vec.Text8Corpus` ，
   `gensim.models.word2vec.LineSentence`

```python
# 使用LineSentence() 
sentences = LineSentence('a.txt')   #  文本格式是 单词空格分开，一行为一个文档
# 使用Text8Corpus() 
sentences = Text8Corpus('a.txt')   #  文本格式是 单词空格分开，一行为一个文
model = Word2Vec(sentences, min_count=1)  # 执行这一句的时候就是在训练模型了
```



#### 模型使用

##### 初始化模型

```python
model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
```

##### 保存加载模型

```python
model.save(fname)
model = Word2Vec.load(fname)  # you can continue training with the loaded model!
```

词向量存储在model.wv的KeyedVectors实例中，可以直接在KeyedVectors中查询词向量。

```python
model.wv['computer']  # numpy vector of a word
array([-0.00449447, -0.00310097,  0.02421786, ...], dtype=float32)
```

词向量也可以被硬盘上已有的C格式文件实例化成KeyedVectors

```python
from gensim.models import KeyedVectors
word_vectors = KeyedVectors.load_word2vec_format('/tmp/vectors.txt', binary=False)  # C text format
word_vectors = KeyedVectors.load_word2vec_format('/tmp/vectors.bin', binary=True)  # C binary format
```

##### 读取vocabulary

```python
list(w2v_model.wv.vocab)[:5]
# ['K H 装', '机 一直 ', '偶发性 报', ' 错误 两', '台 质谱 ']
```

##### 词语相似度计算

```python
model.most_similar(positive=['woman', 'king'], negative=['man'])
#输出[('queen', 0.50882536), ...]

model.wv.most_similar_cosmul(positive=['woman', 'king'], negative=['man'])
#输出[('queen', 0.71382287), ...]

model.doesnt_match("breakfast cereal dinner lunch".split())
#输出'cereal'

model.similarity('woman', 'man')
#输出0.73723527
```

##### 模型下的文本概率

```python
model.score(["The fox jumped over a lazy dog".split()])
# 0.2158356
```

如果模型训练完成(不再更新)，可以在wv中转换gensim.models.KeyedVectors实例来避免不必要的内存消耗

```python
word_vectors = model.wv
del model
```

gensim.models.phrases模块可以让你自动检测短语的词向量

```python
bigram_transformer = gensim.models.Phrases(sentences)
model = Word2Vec(bigram_transformer[sentences], size=100, ...)
```

##### 估算模型所需内存

estimate_memory（vocab_size = None，report = None ）
使用当前设置和提供的词汇大小估算模型所需的内存。





### 参考资料

[word2vec参数理解](https：//www.cnblogs.com/kjkj/p/9825418.html)

[用gensim学习word2vec](https://www.cnblogs.com/pinard/p/7278324.html)

[word2vec原理(一) CBOW与Skip-Gram模型基础](https://www.cnblogs.com/pinard/p/7160330.html)

[基于gensim的word2vec实战](https://www.jianshu.com/p/5f04e97d1b27)

[gensim中word2vec使用](https://blog.csdn.net/u010700066/article/details/83070102)

[gensim-word2vec](https://www.jianshu.com/p/0702495e21de)


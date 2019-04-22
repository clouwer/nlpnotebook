## easyNER



搭建一个方便业务工作使用的任务指向型端到端NER框架



### 项目说明

实现从最原始的业务数据到词性标注，到自定义词库，到任务定义，到NER样本转化到模型建立最终demo实现的一整套NER框架。



### 数据说明

使用的demo数据是不方便公开的CCKS2017电子病历标注的数据集，在原数据集的基础上做了一小部分的处理，由于原始的数据集是已经完全POS标注的数据，而真实的业务场景下我们只有原始文本和人工NER标签信息，所以对完全POS的数据进行了部分提取，删去了各词的位置信息只留下了POS标签的信息，只保留了原始的文本信息和POS标签分类，恢复了业务场景下常出现的状态。（对应的数据在`source/data/`下）



#### POS标签

```
右髋部	身体部位
疼痛	症状和体征
肿胀	症状和体征
鼻部	身体部位
面部	身体部位
痛	症状和体征
腰部	身体部位
头晕	症状和体征
上腹部	身体部位
腰背部	身体部位
```

pos标签由人工标注，针对原始数据`txtoriginal`，列出其中我们感兴趣的部分，也是NER的目标。

值得注意的是，由于电子病历文本来源于医生的记录，所以存在同义由多相近词表达的情况，如例中的 **痛** 与 **疼痛**。这些差别可能来源于场景，可能来源于医生的个人习惯。所以在用代码利用pos标签对所有 `txtoriginal` 文本进行标注的时候，需要进行人工的调整，在相似词之间确定优先级。

```python
from pyhanlp import *

f = open(config.userdict, encoding = 'UTF-8')

ner_pos = {}
for line in f:
    res = line.strip().split('	')
    word = res[0]
    nature = res[1]
    ner_pos[word] = nature

f = open('../source/data/ner_data/synonym.txt', 'w+', encoding = 'UTF-8')
CoreSynonymDictionary = JClass("com.hankcs.hanlp.dictionary.CoreSynonymDictionary")
word_array = [i for i in ner_pos.keys()]
similarity = {}
for a in word_array:
    similarity[a] = []
    for b in word_array[word_array.index(a)+ 1:]:
        if CoreSynonymDictionary.similarity(a, b) == 1.0:
            f.write(a + '\t' + b + '\n')
f.close()
```

尝试使用hanlp的语义相似性进行度量，[保存结果](source/data/ner_data/synonym.txt)

可以看到语义相近的结果并不多，如果设定阈值在0.999999而不是1.0，又会有不是同类的词如支气管炎和甲状腺癌被归为同义的情况。在语义相近词并不多的情况下，决定用代码标签后的部分修正来解决问题。
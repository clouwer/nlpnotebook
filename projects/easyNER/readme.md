## easyNER



搭建一个方便业务工作使用的任务指向型端到端NER框架



### 项目说明

实现从最原始的业务数据到词性标注，到自定义词库，到任务定义，到NER样本转化到模型建立最终demo实现的一整套NER框架。



### 数据说明

使用的demo数据是不方便公开的CCKS2017电子病历标注的数据集，在原数据集的基础上做了一小部分的处理，由于原始的数据集是已经完全POS标注的数据，而真实的业务场景下我们只有原始文本和人工NER标签信息，所以对完全POS的数据进行了部分提取，删去了各词的位置信息只留下了POS标签的信息，只保留了原始的文本信息和POS标签分类，恢复了业务场景下常出现的状态。（对应的数据在`source/data/`下）



#### POS标签

##### 目标序列标记

- O非实体部分
- TREATMENT治疗方式,
- BODY身体部位,
- SIGN疾病症状,
- CHECK医学检查, 
- DISEASE疾病实体,

##### 序列标记方法：采用BIO三元标记

```python
class_dict ={
    'O':0,
    'B-TREATMENT': 1,
    'I-TREATMENT': 2,
    'B-BODY': 3,
    'I-BODY': 4,
    'B-SIGNS': 5,
    'I-SIGNS': 6,
    'B-CHECK': 7,
    'I-CHECK': 8,
    'B-DISEASE': 9,
    'I-DISEASE': 10
}
```

pos标签由人工标注，针对原始数据`txtoriginal`，列出其中我们感兴趣的部分，也是NER的目标。

值得注意的是，由于电子病历文本来源于医生的记录，所以存在同义由多相近词表达的情况，如例中的 **痛** 与 **疼痛**。这些差别可能来源于场景，可能来源于医生的个人习惯。所以在用代码利用pos标签对所有 `txtoriginal` 文本进行标注的时候，需要进行人工的调整，在相似词之间确定优先级。

以第一条记录举例

```
1.患者老年女性，88岁；2.既往体健，否认药物过敏史。3.患者缘于5小时前不慎摔伤，伤及右髋部。伤后患者自感伤处疼痛，呼我院120接来我院，查左髋部X光片示：左侧粗隆间骨折。给予补液等对症治疗。患者病情平稳，以左侧粗隆间骨折介绍入院。患者自入院以来，无发热，无头晕头痛，无恶心呕吐，无胸闷心悸，饮食可，小便正常，未排大便。4.查体：T36.1C，P87次/分，R18次/分，BP150/93mmHg,心肺查体未见明显异常，专科情况：右下肢短缩畸形约2cm，右髋部外旋内收畸形，右髋部压痛明显，叩击痛阳性,右髋关节活动受限。右足背动脉波动好，足趾感觉运动正常。5.辅助检查：本院右髋关节正位片：右侧股骨粗隆间骨折。
```

可以看到文中存在很多组相似词覆盖的现象：

| word       | nature     | word           | nature     |
| ---------- | ---------- | -------------- | ---------- |
| 头         | 身体部位   | 头痛           | 症状和体征 |
| 晕         | 症状和体征 | 头晕           | 症状和体征 |
| 左侧粗隆间 | 身体部位   | 左侧粗隆间骨折 | 疾病和诊断 |
| X光片      | 检查和检验 | 左髋部X光片    | 检查和检验 |
| 。。。     |            |                |            |

考虑采取优先取长词的思路，此时需注意的是，有可能存在同一句子中既有`X光片`又有`左髋部X光片`的情况。也存在同一句中存在单词多次出现的情况。

```python
import re

f = open('../source/data/ner_data/ner_train.txt', 'w+',encoding = 'UTF-8')
for root,dirs,files in os.walk(config.origin_datapath):
    for file in files:
        filepath = os.path.join(root, file)    
        res_dict = {}
        text = re.sub('\n', '', open(filepath, encoding = 'UTF-8').read())
        sg_word = sorted([word for word in ner_pos.keys() if word in text],key = lambda i:len(i),reverse=False)
        sg_words = []
        for word in sg_word:
            sg_words = sg_words + [word] * text.count(word)
        sg_nature = [ner_pos[word] for word in sg_words]
        words_begin = []; words_end = []
        for word in sg_word:
            if word in ['胸部正位+左斜位片','A+B']:
                word = word[:word.index('+')] + '\\'+ word[word.index('+'):]
            words_begin = words_begin + [m.start() for m in re.finditer(word, text)]
            words_end = words_end + [m.end()-1 for m in re.finditer(word, text)]
        res_dict = {}
        for w in range(len(sg_words)):
            res = sg_words[w]
            start = words_begin[w]
            end = words_end[w]
            label = config.label_dict[sg_nature[w]]
            for i in range(start, end+1):
                if i == start:
                    label_cate = 'B-' + label
                else:
                    label_cate = 'I-' + label
                res_dict[i] = label_cate
        for indx, char in enumerate(text):
            char_label = res_dict.get(indx, 'O')
            f.write(char + '\t' + char_label + '\n')
f.close()
```

[训练数据集](source/data/ner_data/ner_train.txt)生成在`source/data/ner_data/ner_train.txt` 

```python
# 正则化找不到带+号的文本
re.finditer('胸部正位+左斜位片', text)
```



> 中文NER训练数据集的样本量不是由数据文本条数决定的，是有句子数量决定的，句子划分标识为['。','?','!','！','？']



### 模型训练

NER部分采用当前state-of-the-art对应的BiLSTM+CRF方案和BERT方案



#### BiLSTM+CRF

运行`src/lstm_train.py` ，模型保存在 `src/model` 目录下

模型设置input_length = 150，所以预测之前要把长文本打散成150字以下的短句。

可使用[预训练词向量](https://github.com/Embedding/Chinese-Word-Vectors)，下载修改代码的embeding部分代替`src/model`目录下的`token_vec_300.bin`即可

| 模型       | 训练集 | 测试集 | 训练集准确率 | 测试集准确率 |          |
| ---------- | ------ | ------ | ------------ | ------------ | -------- |
| BiLSTM+CRF | 6268   | 1571   | 0.9984       | 0.8895       | 8个EPOCH |

2, 模型的测试: python lstm_predict.py, 对训练好的实体识别模型进行测试,测试效果如下:



##### 输出结果

```
[('伤', 'O'), ('后', 'O'), ('患', 'O'), ('者', 'O'), ('自', 'O'), ('感', 'O'), ('伤', 'O'), ('处', 'O'), ('疼', 'B-SIGNS'), ('痛', 'I-SIGNS'), ('，', 'O'), ('呼', 'O'), ('我', 'O'), ('院', 'O'), ('1', 'O'), ('2', 'O'), ('0', 'O'), ('接', 'O'), ('来', 'O'), ('我', 'O'), ('院', 'O'), ('，', 'O'), ('查', 'O'), ('左', 'B-CHECK'), ('髋', 'I-CHECK'), ('部', 'I-CHECK'), ('X', 'I-CHECK'), ('光', 'I-CHECK'), ('片', 'I-CHECK'), ('示', 'O'), ('：', 'O'), ('左', 'B-BODY'), ('侧', 'I-BODY'), ('粗', 'I-DISEASE'), ('隆', 'I-DISEASE'), ('间', 'I-DISEASE'), ('骨', 'I-DISEASE'), ('折', 'I-DISEASE'), ('。', 'O'), ('给', 'O'), ('予', 'O'), ('补', 'O'), ('液', 'O'), ('等', 'O'), ('对', 'O'), ('症', 'O'), ('治', 'O'), ('疗', 'O'), ('。', 'O'), ('患', 'O'), ('者', 'O'), ('病', 'O'), ('情', 'O'), ('平', 'O'), ('稳', 'O'), ('，', 'O'), ('以', 'O'), ('左', 'B-DISEASE'), ('侧', 'I-DISEASE'), ('粗', 'I-DISEASE'), ('隆', 'I-DISEASE'), ('间', 'I-DISEASE'), ('骨', 'I-DISEASE'), ('折', 'I-DISEASE'), ('介', 'O'), ('绍', 'O'), ('入', 'O'), ('院', 'O'), (' 。', 'O')]
```



#### BERT

bert部分暂时实现调用已有的Kashgari包

与BiLSTM-CRF GPU占用36%相比，Fine-Tune BERT时GPU占用轻松达到100%

| 模型 | 训练集 | 测试集 | 训练集准确率 | 测试集准确率 |            |
| ---- | ------ | ------ | ------------ | ------------ | ---------- |
| BERT | 6268   | 1571   | 0.9983       | 0.9905       | 100个EPOCH |

在`src`下运行 `python bert_ner.py --option train` 训练，运行`python bert_ner.py` 进行测试

##### 输出结果

```
[('1', 'O'), ('.', 'O'), ('患', 'O'), ('者', 'O'), ('老', 'O'), ('年', 'O'), ('女', 'O'), ('性', 'O'), ('，', 'O'), ('8', 'O'), ('8', 'O'), ('岁', 'O'), ('；', 'O'), ('2', 'O'), ('.', 'O'), ('既', 'O'), ('往', 'O'), ('体', 'B-BODY'), ('健', 'I-BODY'), ('，', 'O'), ('否', 'O'), ('认', 'O'), ('药', 'B-TREATMENT'), ('物', 'I-TREATMENT'), ('过', 'O'), ('敏', 'O'), ('史', 'O'), ('。', 'O'), ('3', 'O'), ('.', 'O'), ('患', 'O'), ('者', 'O'), ('缘', 'O'), ('于', 'O'), ('5', 'O'), (' 小', 'O'), ('时', 'O'), ('前', 'O'), ('不', 'O'), ('慎', 'O'), ('摔', 'O'), ('伤', 'O'), ('，', 'O'), ('伤', 'O'), ('及', 'O'), ('右', 'B-BODY'), ('髋', 'I-BODY'), ('部', 'I-BODY'), ('。', 'O'), ('伤', 'O'), ('后', 'O'), ('患', 'O'), ('者', 'O'), ('自', 'O'), ('感', 'O'), ('伤', 'O'), ('处', 'O'), ('疼', 'B-SIGNS'), ('痛', 'I-SIGNS'), ('，', 'O'), ('呼', 'O'), ('我', 'O'), ('院', 'O'), ('1', 'O'), ('2', 'O'), ('0', 'O'), ('接', 'O'), ('来', 'O'), ('我', 'O'), ('院', 'O'), (' ，', 'O'), ('查', 'O'), ('左', 'B-CHECK'), ('髋', 'I-CHECK'), ('部', 'I-CHECK'), ('X', 'I-CHECK'), ('光', 'I-CHECK'), ('片', 'I-CHECK'), ('示', 'O'), ('：', 'O'), ('左', 'B-BODY'), ('侧', 'I-BODY'), ('粗', 'I-DISEASE'), ('隆', 'I-DISEASE'), ('间', 'I-DISEASE'), ('骨', 'I-DISEASE'), ('折', 'I-DISEASE'), ('。', 'O'), ('给', 'O'), ('予', 'O'), ('补', 'O'), ('液', 'O'), ('等', 'O'), ('对', 'O'), ('症', 'O'), ('治', 'O'), ('疗', 'O'), ('。', 'O'), ('患', 'O'), ('者', 'O'), ('病', 'O'), ('情', 'O'), ('平', 'O'), ('稳', 'O'), ('，', 'O'), ('以', 'O'), ('左', 'B-DISEASE'), ('侧', 'I-DISEASE'), ('粗', 'I-DISEASE'), ('隆', 'I-DISEASE'), ('间', 'I-DISEASE'), ('骨', 'I-DISEASE'), ('折', 'I-DISEASE'), ('介', 'O'), ('绍', 'O'), ('入', 'O'), ('院', 'O'), ('。', 'O'), ('患', 'O'), ('者', 'O'), ('自', 'O'), ('入', 'O'), ('院', 'O'), ('以', 'O'), ('来', 'O'), ('，', 'O'), ('无', 'O'), ('发', 'B-SIGNS'), ('热', 'I-SIGNS'), ('，', 'O'), ('无', 'O'), ('头', 'B-SIGNS'), ('晕', 'I-SIGNS'), ('头', 'B-SIGNS'), ('痛', 'I-SIGNS'), ('，', 'O'), ('无', 'O'), ('恶', 'B-SIGNS'), ('心', 'I-SIGNS'), ('呕', 'B-SIGNS'), ('吐', 'I-SIGNS'), ('，', 'O'), ('无', 'O'), ('胸', 'B-SIGNS'), ('闷', 'I-SIGNS'), ('心', 'B-SIGNS'), ('悸', 'I-SIGNS'), ('，', 'O'), ('饮', 'O'), ('食', 'O'), ('可', 'O'), ('，', 'O'), ('小', 'B-BODY'), ('便', 'I-BODY'), ('正', 'O'), ('常', 'O'), ('，', 'O'), ('未', 'O'), ('排', 'O'), ('大', 'B-BODY'), ('便', 'I-BODY'), ('。', 'O'), ('4', 'O'), ('.', 'O'), ('查', 'B-CHECK'), ('体', 'I-CHECK'), ('：', 'O'), ('T', 'B-CHECK'), ('3', 'O'), ('6', 'O'), ('.', 'O'), ('1', 'O'), ('C', 'O'), ('，', 'O'), ('P', 'B-CHECK'), ('8', 'O'), ('7', 'O'), ('次', 'O'), ('/', 'O'), ('分', 'O'), ('，', 'O'), ('R', 'B-CHECK'), ('1', 'O'), ('8', 'O'), ('次', 'O'), ('/', 'O'), ('分', 'O'), ('，', 'O'), ('B', 'B-CHECK'), ('P', 'I-CHECK'), ('1', 'O'), ('5', 'O'), ('0', 'O'), ('/', 'O'), ('9', 'O'), ('3', 'O'), ('m', 'O'), ('m', 'O'), ('H', 'O'), ('g', 'O'), (',', 'O'), ('心', 'B-BODY'), ('肺', 'B-BODY'), ('查', 'B-CHECK'), ('体', 'I-CHECK'), ('未', 'O'), ('见', 'O'), ('明', 'O'), ('显', 'O'), ('异', 'O'), ('常', 'O'), ('，', 'O'), ('专', 'O'), ('科', 'O'), ('情', 'O'), ('况', 'O'), ('：', 'O'), ('右', 'B-BODY'), ('下', 'I-BODY'), ('肢', 'I-BODY'), ('短', 'B-SIGNS'), ('缩', 'I-SIGNS'), ('畸', 'I-SIGNS'), ('形', 'I-SIGNS'), ('约', 'O'), ('2', 'O'), ('c', 'O'), ('m', 'O'), ('，', 'O'), ('右', 'B-BODY'), ('髋', 'I-BODY'), ('部', 'I-BODY'), ('外', 'B-SIGNS'), ('旋', 'I-SIGNS'), ('内', 'I-SIGNS'), ('收', 'I-SIGNS'), ('畸', 'I-SIGNS'), ('形', 'I-SIGNS'), ('，', 'O'), ('右', 'B-BODY'), ('髋', 'I-BODY'), ('部', 'I-BODY'), ('压', 'B-CHECK'), ('痛', 'I-CHECK'), ('明', 'O'), ('显', 'O'), ('，', 'O'), ('叩', 'B-CHECK'), ('击', 'I-CHECK'), ('痛', 'I-CHECK'), ('阳', 'O'), ('性', 'O'), (',', 'O'), ('右', 'B-BODY'), ('髋', 'I-BODY'), ('关', 'I-BODY'), ('节', 'I-BODY'), ('活', 'O'), ('动', 'O'), ('受', 'O'), ('限', 'O'), ('。', 'B-BODY'), ('右', 'I-BODY'), ('足', 'I-BODY'), ('背', 'I-BODY'), ('动', 'I-BODY'), ('脉', 'O'), ('波', 'O'), ('动', 'O'), ('好', 'O'), ('，', 'B-BODY'), ('足', 'I-BODY'), ('趾', 'O'), ('感', 'O'), ('觉', 'O'), ('运', 'O'), ('动', 'O'), ('正', 'O'), ('常', 'O'), ('。', 'O'), ('5', 'O'), ('.', 'O'), ('辅', 'O'), ('助', 'B-CHECK'), ('检', 'I-CHECK'), ('查', 'O'), ('：', 'O'), ('本', 'O'), ('院', 'B-CHECK'), ('右', 'I-CHECK'), ('髋', 'I-CHECK'), ('关', 'I-CHECK'), ('节', 'I-CHECK'), ('正', 'I-CHECK'), ('位', 'I-CHECK'), ('片', 'O'), ('：', 'B-DISEASE'), ('右', 'I-BODY'), ('侧', 'I-BODY'), ('股', 'I-BODY'), ('骨', 'I-DISEASE'), ('粗', 'I-DISEASE'), ('隆', 'I-DISEASE'), ('间', 'I-DISEASE'), ('骨', 'I-DISEASE'), ('折', 'O')]
```

> bert的测试结果看到在250字以上出现结果错位情况，代码有部分bug后续工作修复



## References

https://github.com/liuhuanyong/MedicalNamedEntityRecognition

https://github.com/Hironsan/anago

https://github.com/liuhuanyong/ChineseHumorSentiment

https://github.com/BrikerMan/Kashgari
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



### 模型训练

NER部分采用当前state-of-the-art对应的BiLSTM+CRF方案和BERT方案

#### BiLSTM+CRF

运行`src/lstm_train.py` ，模型保存在 `src/model` 目录下

```shell
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding_1 (Embedding)      (None, 150, 300)          527700
_________________________________________________________________
bidirectional_1 (Bidirection (None, 150, 256)          439296
_________________________________________________________________
dropout_1 (Dropout)          (None, 150, 256)          0
_________________________________________________________________
bidirectional_2 (Bidirection (None, 150, 128)          164352
_________________________________________________________________
dropout_2 (Dropout)          (None, 150, 128)          0
_________________________________________________________________
time_distributed_1 (TimeDist (None, 150, 11)           1419
_________________________________________________________________
crf_1 (CRF)                  (None, 150, 11)           275
=================================================================
Total params: 1,133,042
Trainable params: 605,342
Non-trainable params: 527,700
_________________________________________________________________
Train on 6268 samples, validate on 1568 samples
Epoch 1/8
6268/6268 [==============================] - 69s 11ms/step - loss: 18.5580 - crf_viterbi_accuracy: 0.6915 - val_loss: 15.8169 - val_crf_viterbi_accuracy: 0.7778
Epoch 2/8
6268/6268 [==============================] - 66s 10ms/step - loss: 17.8016 - crf_viterbi_accuracy: 0.9259 - val_loss: 15.5995 - val_crf_viterbi_accuracy: 0.8267
Epoch 3/8
6268/6268 [==============================] - 65s 10ms/step - loss: 17.6863 - crf_viterbi_accuracy: 0.9601 - val_loss: 15.5239 - val_crf_viterbi_accuracy: 0.8432
Epoch 4/8
6268/6268 [==============================] - 66s 11ms/step - loss: 17.6465 - crf_viterbi_accuracy: 0.9720 - val_loss: 15.4771 - val_crf_viterbi_accuracy: 0.8546
Epoch 5/8
6268/6268 [==============================] - 71s 11ms/step - loss: 17.6257 - crf_viterbi_accuracy: 0.9783 - val_loss: 15.4423 - val_crf_viterbi_accuracy: 0.8657
Epoch 6/8
6268/6268 [==============================] - 72s 12ms/step - loss: 17.6131 - crf_viterbi_accuracy: 0.9824 - val_loss: 15.4186 - val_crf_viterbi_accuracy: 0.8724
Epoch 7/8
6268/6268 [==============================] - 149s 24ms/step - loss: 17.6047 - crf_viterbi_accuracy: 0.9858 - val_loss: 15.4104 - val_crf_viterbi_accuracy: 0.8767
Epoch 8/8
6268/6268 [==============================] - 68s 11ms/step - loss: 17.5979 - crf_viterbi_accuracy: 0.9879 - val_loss: 15.4195 - val_crf_viterbi_accuracy: 0.8797
```

模型设置input_length = 150，所以预测之前要把长文本打散成150字以下的短句。



##### 输出结果

```
[('伤', 'O'), ('后', 'O'), ('患', 'O'), ('者', 'O'), ('自', 'O'), ('感', 'O'), ('伤', 'O'), ('处', 'O'), ('疼', 'B-SIGNS'), ('痛', 'I-SIGNS'), ('，', 'O'), ('呼', 'O'), ('我', 'O'), ('院', 'O'), ('1', 'O'), ('2', 'O'), ('0', 'O'), ('接', 'O'), ('来', 'O'), ('我', 'O'), ('院', 'O'), ('，', 'O'), ('查', 'O'), ('左', 'B-CHECK'), ('髋', 'I-CHECK'), ('部', 'I-CHECK'), ('X', 'I-CHECK'), ('光', 'I-CHECK'), ('片', 'I-CHECK'), ('示', 'O'), ('：', 'O'), ('左', 'B-BODY'), ('侧', 'I-BODY'), ('粗', 'I-DISEASE'), ('隆', 'I-DISEASE'), ('间', 'I-DISEASE'), ('骨', 'I-DISEASE'), ('折', 'I-DISEASE'), ('。', 'O'), ('给', 'O'), ('予', 'O'), ('补', 'O'), ('液', 'O'), ('等', 'O'), ('对', 'O'), ('症', 'O'), ('治', 'O'), ('疗', 'O'), ('。', 'O'), ('患', 'O'), ('者', 'O'), ('病', 'O'), ('情', 'O'), ('平', 'O'), ('稳', 'O'), ('，', 'O'), ('以', 'O'), ('左', 'B-DISEASE'), ('侧', 'I-DISEASE'), ('粗', 'I-DISEASE'), ('隆', 'I-DISEASE'), ('间', 'I-DISEASE'), ('骨', 'I-DISEASE'), ('折', 'I-DISEASE'), ('介', 'O'), ('绍', 'O'), ('入', 'O'), ('院', 'O'), (' 。', 'O')]
```



## References

https://github.com/liuhuanyong/MedicalNamedEntityRecognition

https://github.com/Hironsan/anago

https://github.com/liuhuanyong/ChineseHumorSentiment
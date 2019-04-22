## easyNER



搭建一个方便业务工作使用的任务指向型端到端NER框架



### 项目说明

实现从最原始的业务数据

```
杭州市妇产科医院2号1235仪器，联机报错，1号针注射器工作异常
```

到词性标注，到自定义词库，到任务定义，到NER样本转化到模型建立最终demo实现的一整套NER框架。

模型输出样式：

```
杭 B-OGN
州 I-OGN
市 I-OGN
妇 I-OGN
产 I-OGN
科 I-OGN
医 I-OGN
院 I-OGN
2 O
号 O
1 B-INS
2 I-INS
3 I-INS
5 I-INS
仪 B-INS
器 I-INS
，O
联 B-ERR
机 I-ERR
报 I-ERR
错 I-ERR
，O
1 O
号 O
针 B-INS
注 B-INS
射 I-INS
器 I-INS 
工 B-ERR
作 I-ERR
异 I-ERR
常 I-ERR
```


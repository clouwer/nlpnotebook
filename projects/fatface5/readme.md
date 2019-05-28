## 小肥脸5号



一个聊天机器人的实现~~~~



### 拟实现功能

API部分

- 微信端口连接
- 图灵机器人API
- BosonNLP

数据获取部分

- 豆瓣小组爬虫
- 微信聊天记录语料整理

聊天机器人部分

- NLU，NLG
- 人格信息
- 敏感词过滤
- 斗图



### 产品定位

- 任务驱动式的半闲聊机器人
- 主要工作是夸到开心
- 偶尔还能查查天气，测测运势
- 基于规则的小游戏模块



### 数据获取

使用什么数据集训练决定了对话类型

- 我们需要更多的爬虫
- 夸夸模块对话语料来源：
  - [豆瓣相互表扬小组](https://www.douban.com/group/kuakua/)
- 情侣聊天语料来源：
  - 与女朋友的聊天记录



### 意图识别

由于任务驱动 + 功能局限的特性，实现方案优先考虑基于规则的排序方案

在后续版本中再进行迭代



### 人格信息

- 个人信息：小肥脸5号/性别男/爱好女/20岁/生日5月27日/

- 日常爱好：打篮球/写代码/吃/玩鞋/
- 其他信息：
  - 有个18岁的女朋友/小崽崽7号/性别女/爱好肥脸/生日6月1号/
  - 



### 参考资料

玻森API ：https://bosonnlp.com/dev/center

https://gitbook.cn/books/5a4a16da1f2e8d585e464f44/index.html

http://blog.topspeedsnail.com/archives/10735/comment-page-1#comment-1161%E3%80%82

豆瓣小组爬虫 ：https://github.com/kaito-kidd/douban-group-spider

https://github.com/qhduan/ConversationalRobotDesign

https://github.com/zhihao-chen/QASystemOnMedicalGraph


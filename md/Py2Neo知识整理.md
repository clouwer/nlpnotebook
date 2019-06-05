## Py2Neo知识整理



### Neo4j

Neo4j是一个世界领先的开源图形数据库，由 Java 编写。图形数据库也就意味着它的数据并非保存在表或集合中，而是保存为节点以及节点之间的关系。

Neo4j 的数据由下面几部分构成：

- 节点
- 边
- 属性

Neo4j 除了顶点（Node）和边（Relationship），还有一种重要的部分——属性。无论是顶点还是边，都可以有任意多的属性。属性的存放类似于一个 HashMap，Key 为一个字符串，而 Value 必须是基本类型或者是基本类型数组。

在Neo4j中，节点以及边都能够包含保存值的属性，此外：

- 可以为节点设置零或多个标签（例如 Author 或 Book）
- 每个关系都对应一种类型（例如 WROTE 或 FRIEND_OF）
- 关系总是从一个节点指向另一个节点（但可以在不考虑指向性的情况下进行查询）

具体介绍可以参考：<https://www.w3cschool.cn/neo4j>。



### Py2Neo用法

Py2Neo 是用来对接 Neo4j 的 Python 库，相关链接

- 官方文档：<http://py2neo.org/v3/index.html>
- GitHub：<https://github.com/technige/py2neo>



#### Node & Relationship

Neo4j 里面最重要的两个数据结构就是节点和关系，即 Node 和 Relationship，可以通过 Node 或 Relationship 对象创建



#### Subgraph

Subgraph，子图，是 Node 和 Relationship 的集合，最简单的构造子图的方式是通过关系运算符





### 参考资料

[Neo4j简介及Py2Neo的用法](https://cuiqingcai.com/4778.html)
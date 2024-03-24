## 一、数学/逻辑/比较/优先级运算/方程等字符串表达式 ——> AST语法树解析
### 1. 可返回str类型表达式
### 2. 可遍历AST语法树
### 3. 实现剪枝操作等
### 4. 通过常量赋值/剪枝等，实现变量自动删除
### 5. 与cse算法一起，实现性能迭代  
### ```ast_all.py```实现

## 二、json文件(dict类型)，转GPU的tf图结构
### 1. 基于上述AST语法树，实现转tf图结构（savemodel.pb格式）。
### 2. 包括tf_serving_wamup文件的产出逻辑。 ```ast2tf.py```实现，生成文件至```/savedmodel```
### 3. 本代码主要实现功能，举例较为简单。实际应用场景中实验变量（or 用户/物料）特征极多，本代码已针对该场景实现了拷贝优化等
### 4. 实际应用场景中，用户侧特征（实验变量） 与 物料侧特征（实验变量）的batch_size可能不同。大多数在serving阶段会对用户侧做n合1优化。本代码已针对该类场景实现优化。

## 三、基于AST语法树的CSE优化
### 1. 将公共子表达式提出，减少重复计算；对于满足交换率（如a*b=b*a），本代码已适配
# Topic & 没有Topic对Query的影响


# 生成两种版本的区别
## No Trainer版本和Trainer版本目前的区别
### 语料
* Trainer语料里加截断了
* No Trainer在代码里加了截断 --未检查

### 学习率
* Learning rate Trainer版本 设置成了2e-5
* No Trainer是1e-5

lr = 2e-5效果会更好



### weight decay
* No Trainer版本Weight decay = 0, Warmup_step = 1000
* Trainer版本需要仔细检查一下

### 模型加载
* No Trainer版本只加载10层
* Trainer版本加载了12层encoder但是弃用了最后2层

## 表现差异
两个Sample


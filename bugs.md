# BUGS
## 单卡bug
1. 不能用fp16， `/home2/gaobofei/anaconda3/lib/python3.8/site-packages/accelerate/utils.py`这个路径下的`convert_to_fp32`会把模型输出的一部分删除。

## 多卡bug
1. 不能多卡训练。

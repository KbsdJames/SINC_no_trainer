# DUSINC code
## CPT Code: No Trainer Version
* requirements:
    * 见`requirements.txt`
* 使用Accelerate版本的代码，包含了数据预处理`data_preprocessing_query/response.py`以及训练之前进行特定的预处理（见`query/response_gen_notrainer.py`）。  其中`preprocess_function_not_test`是训练集以及测试集的处理函数，`preprocess_function_test`是测试集的处理函数。
* 训练&预测:
    ~~~
    > mv modeling_cpt.py ../ #modeling_cpt.py原本属于上层目录，上层目录有一些CPT自带的没有用到的文件就没有上传
    > python data_preprocessing_reponse.py
    > bash finetune_dial.sh
    ~~~
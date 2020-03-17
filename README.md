
# 一、项目介绍 
深度学习在ctr预估中的应用，使用tensorflow estimator进行封装，包括dnn、deepfm、din、autoint等模型

该项目针对单目标任务，这里给出多目标工程链接：

# 二、代码目录

##1.主函数

    local_run.py
    local_run_test.py——用于本地测试
    local_run_hours.py——针对小时级更新模型


##2.utils——工具函数文件夹

    utils/my_utils.py——一些基础共用函数
    utils/data_loader.py——加载tfrecord数据
    utils/model_layer.py——模型共用的内部模块，如自定义网络层，多值特征处理等
    utils/model_op.py——模型共用的训练、预测模块


##3.models——即深度学习模型文件，包括模型构建、训练、预测等

    # attention思想
    models/din —— din模型
    models/dien —— dien模型(din+rnn)
    models/dnn_autoint —— autoint，使用self-attention，一种特征处理，可以应用于任何模型上
    models/dinfm —— din与deepfm的结合（微创新）
    
    # dnn、deepfm模型
    models/dnn——原始dnn模型
    models/dnn_cate——仅使用离散特征的dnn模型
    models/dnn_emb——连续特征离散化+离散特征的dnn模型
    models/dnn_pool——对多值离散进行池化的dnn模型
    ...
    models/deepfm.py——....类似上面dnn，此处省略
    models/deepfm_cate.py——...
    ...

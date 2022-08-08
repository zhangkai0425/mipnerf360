import torch
import para

"""
关于regularization.py做如下部署：
1.完成t->s的映射的函数，主要是ray.py中其实这部分也要重新修改，
2.完成w(s)的设计与实现
3.完成loss_regulariation的设计和代码编写
4.跑起来JAX的环境，跑自己的数据得到较好的效果-周三之前必须跑出来结果
5.Pipline跑通-周三之前必须跑通
6.整理实验文档-周五之前必须完成
"""
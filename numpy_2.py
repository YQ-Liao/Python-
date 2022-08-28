import copy
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
import networkx as nx
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.semi_supervised import _label_propagation
import torch.utils.data as Data
import hiddenlayer as hl
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import seaborn as sns
import torch.optim as optim
import torchvision
from torchvision import transforms
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

t1=np.arange(0,12).reshape(2,6)
# print("t1:\n",t1)
t2=np.arange(12,24).reshape(2,6)
# print("t2:\n",t2)

#---np的拼接---
t_v=np.vstack((t1,t2))  #对t1,t2进行竖直拼接，注意t1,t2的顺序
# print("竖直拼接:\n",t_v)
t_h=np.hstack((t1,t2)) #对t1,t2进行水平拼接，注意t1,t2的顺序
# print("水平拼接:\n",t_h)

#---np的交换行或者列---
t1[[0,1],:]=t1[[1,0],:] #把t1的第一行和第二行交换
# print("交换行后的t1:\n",t1)
t1[:,[0,3]]=t1[:,[3,0]] #把第一列和第四列交换
# print("交换列后的t1:\n",t1)

#---numpy生成随机数---
a=np.random.rand(3,4) #输入形状，输出0-1之间均匀分布的浮点数
# print(a)
b=np.random.randn(3,4) #输入形状，输出均值为0，方差为1服从正态分布的浮点数
# print(b)
c=np.random.randint(0,5,(3,4)) #输入上下限、形状，输出指定范围内、指定形状的随机整数
print(c)
d=np.random.uniform(0,5,(3,4)) #输入上下限、形状，输出指定范围内、指定形状、满足均匀分布的浮点数
# print(d)
e=np.random.normal(3,10,(3,4)) #输入上下限、形状，输出指定范围内、指定形状、满足正态分布的浮点数
# print(e)

#---numpy的拷贝---
c_1=c #c_1和c指向同一内存，改变c_1同时会改变c，改变c同时会改变c_1
c_2=c.copy() #c_2新开一个内存，c和c_2互不影响

#---numpy常用统计函数---
#求和
# print(c.sum(axis=0)) #按列求和
# print(c.sum(axis=1)) #按行求和

#求均值
# print(c.mean(axis=0)) #按列求均值
# print(c.mean(axis=1)) #按行求均值

#求中位数
# print(np.median(c,axis=0)) #按列求中位数
# print(np.median(c,axis=1)) #按行求中位数

#求最大最小值
# print(c.max(axis=0)) #求列的最大值
# print(c.max(axis=1)) #求行的最大值
# print(c.min(axis=0)) #求列的最小值
# print(c.min(axis=1)) #求行的最小值








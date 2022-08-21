import copy
import numpy as np
import pandas as pd
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

#---创建一个DataFrame，并获取其参数---
data=pd.DataFrame(data=np.arange(24).reshape(4,6),index=list("abcd"),columns=list("WXYZOP"))
print(data) #内容
# print(data.shape) #形状
# print(data.index) #行索引
# print(data.columns) #列索引
# print(data.dtypes) #数据类型
# print(data.values) #值
# print(data.ndim) #维数

# print(data.head(n)) #输出前n行
# print(data.tail(n)) #输出后n行
# print(data.info()) #输出一些描述信息：
# print(data.describe()) #输出一些统计信息：计数、均值、标准差、最大值、最小值、四分位数

#---把DataFrame中的数据进行排序---
data_sorted=data.sort_values(by="W",ascending=False)
# print(data_sorted)

#---对DataFrame进行切片和索引---

#loc输入的是标签
# print(data.loc["a":"c","X":"Z"]) #连续地取
# print(data.loc[["b","c"],["X","Y"]]) #离散地取，要多加方括号
# print(data.loc[["b","c"],:]) #只取行
# print(data.loc[:,["X","Y"]]) #只取列

# #iloc输入的是位置
# print(data.iloc[0:3,1:4]) #连续地取，但iloc写0:n实际上是取0:n-1
# print(data.iloc[[1,2],[1,2]]) #离散地取，要多加方括号

#按条件进行索引
# print(data[data["Y"]>7]) #只能选择列，输出的是符合条件的若干行
# print(data[(data["Y"]>7)&(data["Y"]<15)]) #多个条件，取交集,写的大于，实际上是大于等于
# print(data[(data["Y"]>19)|(data["Z"]<9)]) #多个条件，取并集
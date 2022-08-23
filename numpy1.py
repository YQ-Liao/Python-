import copy
import random

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

#---创建np数组，并修改数据类型---
a=np.array([1,2,3,4,5])
b=np.array(range(1,6),dtype=float)
c=np.arange(1,6)
# print(a.dtype)
# print(b.dtype)
#产生指定位数的小数
d=np.round(random.random(),3)  #小数点后三位
# print(d)

#---查看和修改数组参数---
sample=np.array([[1,2,3,4,5,6],[7,8,9,10,11,12]])
# print(sample.shape)  #查看形状

sample_reshaped1=sample.reshape(3,4) #修改形状
sample_reshaped2=sample.reshape(4,-1) #自适应修改形状，用-1表达
# sample_reshaped3=sample.reshape(-1) #自适应展开成一维,法1
sample_reshaped3=sample.flatten() #自适应展开成一维，法2

# print(sample_reshaped1)
# print(sample_reshaped2)
# print(sample_reshaped3)

#---numpy读取文件---
# DD_url="./data/DD/DD_A.txt"
# DDA_data=np.loadtxt(fname=DD_url,
#                    delimiter=",",
#                     dtype=int)
# print(DDA_data)

#---numpy索引和修改数值---
x=np.random.randint(1,12,size=(3,4))
print(x)
# x[0,0]=20 #修改某一个数值
# print(x)

# print(x[x>6]) #按条件索引

# x[x<4]=20 #按条件修改法1：值小于4的换成20，其余不变
# print(x)
# x=np.where(x<5,0,1) #按条件修改法2：值小于5的换成0，否则换成1
# print(x)
# x=x.clip(4,6) #按条件修改法3：小于4的换成4，大于6的换成6，其余不变
# print(x)

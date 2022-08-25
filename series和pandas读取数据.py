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

#---创建series---
a=pd.Series([1,2,3,4,5]) #创建series
b=pd.Series([1,2,3,4,5],index=list("abcde")) #创建series时指定index,也可以用来修改已经存在的series的index
c_dict={"name":"X","age":24,"tel":114514} #通过字典创建series
c_series=pd.Series(c_dict)
# print(c_series)

#---取series的元素---
# print(c_series[1]) #可以看到只会输出内容而不会输出索引，连续取、离散取、条件取和DataFrame同样用法
# print(c_series.index) #输出series的索引
# print(list(c_series.index)) #转换为list类型
# print(c_series.values) #输出series的元素值

#---pandas读取文件---
url="./data/example.csv"
example=pd.read_csv(url) #pandas读取csv文件。除此之外还可以read_excel、read_html、read_json、read_sql...
print(example.dtypes)



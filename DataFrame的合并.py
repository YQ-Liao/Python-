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

#---DataFrame的合并---
data1=pd.DataFrame(data=np.arange(24).reshape(4,6),index=list("abcd"),columns=list("WXYZOP"))
# print("data1:\n",data1)
data2=pd.DataFrame(data=np.arange(24).reshape(4,6),index=list("abcd"),columns=list("012345"))
# print("data2:\n",data2)
#使用join合并
data_join=data1.join(data2) #用join按行索引合并，把行索引相同的DataFrame合并，注意顺序
# print("data_join:\n",data_join)
#使用merge合并
# merge合并需要指定某一列
# merge有4种合并方式：inner（交集）,outter（并集）,right（左合并）,left（右合并）  默认inner
#使用方式为:(1)列名相同的情况：df1.merge(df2,on="列名",how="合并方式")
#         (2)列名不同的情况：df1.merge(df2,left_on="左列名",right_on="右列名",how="合并方式")



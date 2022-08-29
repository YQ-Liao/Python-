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
data1.loc[:,"X"]=["cn","cn","jp","kr"]
print(data1)
grouped=data1.groupby(by="X")  #按照X中的元素进行分组
# for i,j in grouped:  #i是元素,j是DataFrame
#     print("-"*10)
#     print(i)
#     print("*"*10)
#     print(j)
print(grouped.count()) #统计元素个数



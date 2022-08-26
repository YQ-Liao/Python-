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


url_1="./data/example_1.csv"
example_1=pd.read_csv(url_1)
print("科目:\n",example_1["科目"])

temp=example_1["科目"].str.split(",").tolist()
temp_list=[]
for i in temp:
    temp_list+=i[0].split('，')   #可以直接相加来合并集合
temp_setlist=list(set(temp_list)) #得到用set去重后的列表
print("temp:\n",temp)
print("temp_setlist:\n",temp_setlist)

mask_np=np.zeros([example_1.shape[0],len(temp_setlist)]) #设置全0的np，形状用参数设置
mask_df=pd.DataFrame(mask_np,columns=temp_setlist) #先设置一个全0的DataFrame
print("zero_mask_df:\n",mask_df)

for i in range(example_1.shape[0]):
    mask_df.loc[i,temp[i][0].split("，")]=1  #对应位置赋值,这是重点
print("mask_df:\n",mask_df)





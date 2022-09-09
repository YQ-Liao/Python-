import torch
import torch.utils.data as Data

#---把数据打包成数据集---
# class Mydataset(Data.Dataset):                                   #把numpy数组打包成数据集
#     def __init__(self,x,y):
#         self.x =x
#         self.y =y
#     def __getitem__(self,index):
#         return self.x[index], self.y[index]                      #注意要返回这样的二元组
#     def __len__(self):
#         return len(self.y)                                       #因为例子是numpy形式，所以用len，如果是张量，可以用size或shape
#
# x=np.array([[1, 3], [5, 7], [0, 2], [4, 5], [7, 10]])            #此处是随便设置的例子，实际数据可以从文件中读取
# y=np.array([0, 2, 3, 1, 5])
# a=Mydataset(x,y)                                                 #类实例化
# # print(a[1])
# # print(a.__len__())

x=torch.rand(size=(20,1))
y=torch.randint(0,9,size=(20,1))
tensor_dataset=Data.TensorDataset(x,y)                             #把张量打包成数据集
# print(tensor_dataset[1])

#---为数据集设置加载器---
myLoader=Data.DataLoader(dataset=tensor_dataset,batch_size=5,shuffle=True,num_workers=0)

for step,(batch_x,batch_y) in enumerate(myLoader):                #batch_x,batch_y别忘了加括号
    print("step:\n",step)
    print("batch_x:\n",batch_x)
    print("batch_y:\n",batch_y)
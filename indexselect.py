import torch
import numpy as np
import os
from PIL import Image, TarIO
import pickle
import tarfile
from torch.utils import data

t = torch.arange(24).reshape(2, 3, 4) # 初始化一个tensor，从0到23，形状为（2,3,4）
print("t--->", t)
 
index = torch.tensor([2, 1, 0, 3]) # 要选取数据的位置
print("index--->", index)
 
# data1 = t.index_select(1, index) # 第一个参数:从第1维挑选， 第二个参数:从该维中挑选的位置
# print("data1--->", data1)
 
data2 = t.index_select(2, index) # 第一个参数:从第2维挑选， 第二个参数:从该维中挑选的位置
print("data2--->", data2)

x_cat = t

for k in range(2):
    idx_ = torch.tensor([k],dtype=torch.long)
    # index_swap_per = torch.index_select(index_swap,dim = 0, index= idx_)
    swap_select_per = torch.index_select(t, dim = 0, index= idx_)
    
    b = torch.index_select(swap_select_per, 2, index)
    
    print("b--->", b) 
    
    x_cat = torch.cat( (x_cat,b),0 )
    
print("x_cat--->", x_cat)
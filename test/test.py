# 分类
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

a = [1, 2, 3]
b = ['a', 'b', 'c']
c = [10, 20, 30]

z = zip(a, b, c)

for x, y, z in z:
    print(x, y, z)

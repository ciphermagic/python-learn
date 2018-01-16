# 激励函数
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# 假数据
x = torch.linspace(-5, 5, 200)
# 转为 variable
x = Variable(x)
# x 轴转为 numpy
x_np = x.data.numpy()

# 各种激励函数
y_relu = F.relu(x).data.numpy()
y_sigmoid = F.sigmoid(x).data.numpy()
y_tanh = F.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()
y_softmax = F.softmax(x).data.numpy()

# 画图
plt.figure(figsize=(10, 6))

# 定义图的行列和位置
plt.subplot(231)
# x，y 轴
plt.plot(x_np, y_relu, label='relu')
# 标签位置
plt.legend(loc='best')

plt.subplot(232)
plt.plot(x_np, y_sigmoid, label='sigmoid')
plt.legend(loc='best')

plt.subplot(233)
plt.plot(x_np, y_tanh, label='tanh')
plt.legend(loc='best')

plt.subplot(234)
plt.plot(x_np, y_softplus, label='softplus')
plt.legend(loc='best')

plt.subplot(235)
plt.plot(x_np, y_softmax, label='softmax')
plt.legend(loc='best')

plt.show()

# 回归
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# 我们创建一些假数据来模拟真实的情况.
# 比如一个一元二次函数: y = a * x^2 + b,
# 我们给 y 数据加上一点噪声来更加真实的展示它

# 假数据
x = torch.linspace(-1, 1, 100)
# print(x)

# 一维变二维
x = torch.unsqueeze(x, dim=1)
# print(x)

# x 的平方，加上随机数
y = x.pow(2) + 0.2 * torch.rand(x.size())
# print(x.size())
# print(torch.rand(x.size()))
# print(y)

# 变成 variable
x, y = Variable(x), Variable(y)


# 数据散点图
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

# 建立神经网络
# 建立一个神经网络我们可以直接运用 torch 中的体系.
# 先定义所有的层属性(__init__()), 然后再一层层搭建(forward(x))层于层的关系链接.
# 建立关系的时候, 我们会用到激励函数
class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()  # 继承 __init__ 功能
        # 定义每层用什么样的形式
        # nn.Linear表示的是 y=w*x+b
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output)  # 输出层线性输出

    def forward(self, x):  # 这同时也是 Module 中的 forward 功能
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))  # 激励函数(隐藏层的线性值)
        x = self.predict(x)  # 输出值
        return x


net = Net(n_feature=1, n_hidden=10, n_output=1)

# 快速搭建：
# net = torch.nn.Sequential(
#     torch.nn.Linear(1, 10),
#     torch.nn.ReLU(),
#     torch.nn.Linear(10, 1)
# )

# net 的结构
# print(net)

# 训练网络
# optimizer 优化器
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)  # 传入 net 的所有参数, 学习率
loss_func = torch.nn.MSELoss()  # 预测值和真实值的误差计算公式 (均方差)

for t in range(200):
    prediction = net(x)  # 喂给 net 训练数据 x, 输出预测值
    loss = loss_func(prediction, y)  # 计算两者的误差
    optimizer.zero_grad()  # 清空上一步的残余更新参数值
    loss.backward()  # 误差反向传播, 计算参数更新值
    optimizer.step()  # 将参数更新值施加到 net 的 parameters 上

    # 可视化
    if t % 5 == 0:
        # 清空内容
        plt.cla()
        # 原数据的散点图
        plt.scatter(x.data.numpy(), y.data.numpy())
        # 回归曲线
        plt.plot(x.data.numpy(), prediction.data.numpy())
        # 误差
        plt.text(0.5, 0, 'Loss=%.4f\nt=%d' % (loss.data[0], t))
        plt.pause(0.1)

plt.show()

# 回归（保存与提取）
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt


def save(_x, _y):  # 训练并保存网络
    prediction = None

    # 定义网络
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

    # 训练
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()
    for t in range(200):
        prediction = net1(_x)  # 喂给 net 训练数据 _x, 输出预测值
        loss = loss_func(prediction, _y)  # 计算两者的误差
        optimizer.zero_grad()  # 清空上一步的残余更新参数值
        loss.backward()  # 误差反向传播, 计算参数更新值
        optimizer.step()  # 将参数更新值施加到 net 的 parameters 上

    # 出图
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(_x.data.numpy(), _y.data.numpy())
    plt.plot(_x.data.numpy(), prediction.data.numpy(), color='r')

    # 保存整个网络
    torch.save(net1, 'reg_net.pkl')
    # 仅保存网络参数
    torch.save(net1.state_dict(), 'reg_net_params.pkl')


def restore_net(_x, _y):  # 提取网络
    net2 = torch.load('reg_net.pkl')
    prediction = net2(_x)
    # 出图
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(_x.data.numpy(), _y.data.numpy())
    plt.plot(_x.data.numpy(), prediction.data.numpy(), color='r')


def restore_params(_x, _y):  # 提取网络参数
    # 定义网络
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    # 把参数放进网络
    net3.load_state_dict(torch.load('reg_net_params.pkl'))
    prediction = net3(_x)
    # 出图
    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(_x.data.numpy(), _y.data.numpy())
    plt.plot(_x.data.numpy(), prediction.data.numpy(), color='r')


if __name__ == '__main__':
    # 假数据
    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
    y = x.pow(2) + 0.2 * torch.rand(x.size())
    x, y = Variable(x), Variable(y)

    plt.figure(1, figsize=(10, 3))

    save(x, y)
    restore_net(x, y)
    restore_params(x, y)

    plt.show()

# 批训练回归函数
import torch
from torch.autograd import Variable
import torch.utils.data as Data
import matplotlib.pyplot as plt

# 批训练的数据个数
BATCH_SIZE = 10

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2 * torch.rand(x.size())

# 先转换成 torch 能识别的 Dataset
torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)

# 把 dataset 放入 dataloader
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

net = torch.nn.Sequential(
    torch.nn.Linear(1, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1)
)

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_func = torch.nn.MSELoss()

if __name__ == '__main__':
    for epoch in range(100):
        # 每一步 loader 释放一小批数据用来学习
        for step, (_x, _y) in enumerate(loader):

            b_x = Variable(_x)  # batch x
            b_y = Variable(_y)  # batch y

            prediction = net(b_x)
            loss = loss_func(prediction, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 出图
            plt.clf()
            plt.scatter(x.numpy(), y.numpy())
            a = b_x.data.numpy().squeeze()
            b = prediction.data.numpy().squeeze()
            plt.scatter(b_x.data.numpy().squeeze(), prediction.data.numpy().squeeze(), color='red')
            plt.text(0.5, 0, 'Loss=%.4f\nepoch=%d' % (loss.data[0], epoch))
            plt.pause(0.1)

            if (step + 1) % 10 == 0:
                print("epoch:", epoch, "step:", step, "loss:", loss.data[0])

    plt.show()
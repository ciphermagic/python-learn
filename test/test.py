# 卷积神经网络
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.utils.data as Data

torch.cuda.manual_seed_all(1)

EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False

# Mnist 手写数字
train_data = torchvision.datasets.MNIST(
    root='./mnist',  # 保存或者提取位置
    train=True,
    # 转换 PIL.Image or numpy.ndarray 成
    # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

# 测试数据
test_data = torchvision.datasets.MNIST(root='./mnist', train=False)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1)).type(torch.FloatTensor)[:1000].cuda()
test_y = test_data.test_labels[:1000].cuda()


# 显示手写数字
# plt.imshow(train_data.train_data[0].numpy())
# plt.title('%i' % train_data.train_labels[0])
# plt.show()


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(  # input shape (1, 28, 28)
            torch.nn.Conv2d(
                in_channels=1,  # 输入层
                out_channels=16,  # 输出层
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2
            ),  # output shape (16, 28, 28)
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)  # 在 2x2 空间里向下采样, output shape (16, 14, 14)
        )
        self.conv2 = torch.nn.Sequential(  # input shape (1, 28, 28)
            torch.nn.Conv2d(
                in_channels=16,  # input height
                out_channels=32,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2
            ),  # output shape (32, 14, 14)
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)  # output shape (32, 7, 7)
        )
        self.out = torch.nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output


cnn = CNN().cuda()

# 训练
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = torch.nn.CrossEntropyLoss()  # the target label is not one-hotted

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):  # 分配 batch data, normalize x when iterate train_loader
        b_x = Variable(x).cuda()  # batch x
        b_y = Variable(y).cuda()  # batch y

        output = cnn(b_x)  # cnn output
        loss = loss_func(output, b_y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()

        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data
            accuracy = sum(pred_y == test_y) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)

test_output = cnn(test_x[:10]).cpu()
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10].cpu().numpy(), 'real number')

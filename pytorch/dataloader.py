# 批训练
import torch
import torch.utils.data as Data

# 在训练开始时，参数的初始化是随机的，为了让每次的结果一致，我们需要设置随机种子
torch.manual_seed(1)

# 批训练的数据个数
BATCH_SIZE = 5

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

# 先转换成 torch 能识别的 Dataset
torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)

# 把 dataset 放入 dataloader
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,  # 要不要打乱数据 (打乱比较好)
    num_workers=2,  # 多线程来读数据
)

if __name__ == '__main__':
    # 训练所有整套数据 3 次
    for epoch in range(3):
        # 每一步 loader 释放一小批数据用来学习
        for step, (batch_x, batch_y) in enumerate(loader):
            # 假设这里就是你训练的地方...

            # 打出来一些数据
            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
                  batch_x.numpy(), '| batch y: ', batch_y.numpy())

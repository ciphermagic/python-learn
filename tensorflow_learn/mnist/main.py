# 前馈神经网络
# http://www.360doc.com/content/16/1024/10/37564104_600912207.shtml
# http://blog.topspeedsnail.com/archives/10377
# https://www.leiphone.com/news/201705/TMsNCqjpOIfN3Bjr.html
import os
import random
import struct
import array
import numpy as np
import PIL
import pylab
import pickle


# 前馈神经网络
class NeuralNet(object):
    # 初始化神经网络
    def __init__(self, sizes):
        # sizes 是一个列表，它包含了相关层的神经元个数。
        # 例如，如果我们想要创建一个第一层含有 3 个神经元，第二层含有 4 个神经元，最后一层含有 2 个神经元的 Network 对象，
        # 那么 sizes 为 [3, 4, 2]
        self.sizes_ = sizes

        # 有几层神经网络
        self.num_layers_ = len(sizes)

        # w_、b_初始化为正态分布随机数
        # np.random.randn(y, x) 表示取 y 行 x 列的正太随机数
        # zip: http://www.cnblogs.com/frydsh/archive/2012/07/10/2585370.html

        # 权值：以 [3, 4, 2] 为例，有三层网络，就需要两组权值，分别是第一层到第二层的权值，第二层到第三层的权值
        # 第一层有 3 个神经元，第二层有 4 个神经元，那么第一层到第二层就需要取 4 行 3 列的随机数作为权值，如此类推
        # 这里首先使用 sizes[:-1], sizes[1:]，把 sizes 分为两个列表 [3, 4] 和 [4, 2]
        # 再使用 zip 把两个列表并行，就可以得到前一层到后一层的元组表示：(3, 4)，(4, 2)
        # 即分别需要取 4 行 3 列， 2 行 4 列的随机数作为权值
        self.w_ = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        # 偏置值：除了第一层外，后面有多少层，就需要多少组偏置值，每一组偏置值的数量为该层神经元的个数
        # 例如：第二层是 4，就取 4 个随机数作为偏置值，第三层是 2，就取 2 个随机数作为偏置值
        self.b_ = [np.random.randn(y, 1) for y in sizes[1:]]

    # Sigmoid 函数，S 型曲线，
    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    # Sigmoid 函数的导数
    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    # 给神经网络的输入 x，输出对应的值
    def feedforward(self, x):
        # 前向传输计算每个神经元的值
        for b, w in zip(self.b_, self.w_):
            # 加权求和以及加上 biase
            x = self.sigmoid(np.dot(w, x) + b)
        return x

    # 随机梯度下降
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
           随机梯度下降
           :param training_data: 输入的训练集 (x, y)，代表训练的输入和对应的输出;
           :param epochs: 迭代次数
           :param mini_batch_size: 次训练样本数
           :param eta: 学习率 
           :param test_data: 测试数据集，每次迭代后，打印出正确的结果数
        """
        for j in range(epochs):
            # 搅乱训练集，让其排序顺序发生变化
            random.shuffle(training_data)
            # 分成合适大小的 mini-batches
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                # 根据每个小样本来更新 w 和 b
                self.update_mini_batch(mini_batch, eta)
            # 输出每轮结束后，神经网络的准确度
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), len(test_data)))
            else:
                print("Epoch {0} complete".format(j))

    # 计算损失函数的梯度，然后更新权值以及偏置值
    def update_mini_batch(self, mini_batch, eta):
        # 根据 b 和 w 的行列数创建对应的全部元素值为 0 的空矩阵
        nabla_b = [np.zeros(b.shape) for b in self.b_]
        nabla_w = [np.zeros(w.shape) for w in self.w_]

        for x, y in mini_batch:
            # 根据样本中的每一个输入 x 的其输出 y，计算 w 和 b 的偏导数
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # 计算所有的梯度（偏置值b）的和
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            # 计算所有的另外一个梯度（权值w）的和
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # 根据累加的偏导值更新 w 和 b，这里因为用了小样本，所以 eta 要除于小样本的长度
        self.w_ = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.w_, nabla_w)]
        self.b_ = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.b_, nabla_b)]

    # （核心）反向传播算法
    # 根据输出神经元的值，计算出预测值与真实值之间的误差，
    # 再逆向反馈更新神经网络中每条连接线的权值和每个神经元的偏置值
    def backprop(self, x, y):
        # 根据 b 和 w 的行列数创建对应的全部元素值为 0 的空矩阵
        nabla_b = [np.zeros(b.shape) for b in self.b_]
        nabla_w = [np.zeros(w.shape) for w in self.w_]

        # 前向传输
        activation = x
        # 储存每层的神经元的值的矩阵，下面循环会 append 每层的神经元的值
        activations = [x]
        # 储存每个未经过 sigmoid 计算的神经元的值
        zs = []

        # 从输入层=>隐藏层=>输出层，一层一层的计算所有神经元输出值
        for b, w in zip(self.b_, self.w_):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        # 更新最后一层的权值和偏置值
        # 求 δ 的值
        delta = (activations[-1] - y) * self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        # 乘以前一层的输出值
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # 逆向反馈更新每一层的权值和偏置值
        for l in range(2, self.num_layers_):
            # 从倒数第 l 层开始更新，-l 是 python 中特有的语法表示从倒数第 l 层开始计算
            # 下面这里利用 l+1 层的 δ 值来计算 l 的 δ 值
            delta = np.dot(self.w_[-l + 1].transpose(), delta) * self.sigmoid_prime(zs[-l])
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    # 返回正确识别的个数
    def evaluate(self, test_data):
        # 获得预测结果
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    # 预测
    def predict(self, data):
        value = self.feedforward(data)
        return value.tolist().index(max(value))

    # 保存训练模型
    def save(self):
        # 把 _w 和 _b 保存到文件 (pickle)
        # with open('weight.pickle', 'wb') as f:
        #     pickle.dump(self.w_, f)
        # with open('biase.pickle', 'wb') as f:
        #     pickle.dump(self.b_, f)
        with open('model.pickle', 'wb') as f:
            pickle.dump([self.w_, self.b_], f)

    def load(self):
        # self.w_ = pickle.load(open('weight.pickle', 'rb'))
        # self.b_ = pickle.load(open('biase.pickle', 'rb'))
        model = pickle.load(open('model.pickle', 'rb'))
        self.w_ = model[0]
        self.b_ = model[1]


# 加载 MNIST 数据集
def load_mnist(dataset="training_data", digits=np.arange(10), path="data/"):
    # 读取训练数据
    if dataset == "training_data":
        fname_image = os.path.join(path, 'train-images.idx3-ubyte')
        fname_label = os.path.join(path, 'train-labels.idx1-ubyte')
    # 读取测试数据
    elif dataset == "testing_data":
        fname_image = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_label = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'training_data' or 'testing_data'")

    # 读取 label 二进制文件
    flbl = open(fname_label, 'rb')
    struct.unpack(">II", flbl.read(8))
    lbl = array.array("b", flbl.read())
    flbl.close()

    # 读取图像二进制文件
    fimg = open(fname_image, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = array.array("B", fimg.read())
    fimg.close()

    # 验证 label 的值是否在 0-9 内，只保留符合规范的值
    ind = [k for k in range(size) if lbl[k] in digits]
    # 符合规范的值的长度
    N = len(ind)

    # 创建 N 个 rows 行 cols 列全部为 0 的矩阵
    images = np.zeros((N, rows, cols), dtype=np.uint8)
    # 创建 N 个 1 行 1 列全部为 0 的矩阵
    labels = np.zeros((N, 1), dtype=np.int8)

    # 根据读取文件的值填充矩阵
    for i in range(len(ind)):
        # 填充图像矩阵
        images[i] = np.array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
        # 填充 label 矩阵
        labels[i] = lbl[ind[i]]

    return images, labels


# 加载数据，转换为神经网络需要的格式
def load_samples(dataset="training_data"):
    image, label = load_mnist(dataset)

    # 把 28*28 二维数据转为一维数据
    X = [np.reshape(x, (28 * 28, 1)) for x in image]
    # 灰度值范围 (0-255)，转换为 (0-1)
    X = [x / 255.0 for x in X]

    # 把Y值转换为神经网络的输出格式
    # 把 0-9 的值转换为一维数组，相应位置的值置为 1，其余为 0
    # 例如：5 -> [0,0,0,0,0,1.0,0,0,0]，1 -> [0,1.0,0,0,0,0,0,0,0]
    def vectorized_Y(y):
        e = np.zeros((10, 1))
        e[y] = 1.0
        return e

    # 输入与输出进行配对
    if dataset == "training_data":
        # X：一维图像数组，长度为 60000，每个元素都是一维数组，长度为 28*28=784，
        # 保存的是图像的灰度值转为（0，1）后的表示
        X = X
        # Y：一维 label 数组，长度 60000，每个元素都是一维数组，长度为 10，
        # 保存的是 label 的向量表示，例如：1 -> [0,1.0,0,0,0,0,0,0,0]
        Y = [vectorized_Y(y) for y in label]
        # 通过 zip 操作后，把图像数据与 label 数组进行配对，即对应神经网络的每个输入层和输出层
        pair = list(zip(X, Y))
        return pair
    elif dataset == 'testing_data':
        pair = list(zip(X, label))
        return pair
    else:
        print('Something wrong')


# 把图片转换为一维数组
def load_pic(path):
    # 读取图片,灰度化，并转为数组
    im = pylab.array(PIL.Image.open(path).convert('L'), 'f')
    # 把 28*28 二维数据转为一维数据
    x = np.reshape(im, (28 * 28, 1))
    # 灰度值范围 (0-255)，转换为 (0-1)
    x = x / 255.0
    return x


# 保存或加载模型
def train_model(_net, has_model=False):
    # 如果已经有模型，直接加载模型
    if has_model:
        _net.load()
        print('load model success')
        return
    # 加载数据
    train_set = load_samples(dataset='training_data')
    test_set = load_samples(dataset='testing_data')
    # 开始训练
    _net.SGD(train_set, 30, 10, 3.0, test_data=test_set)
    # 保存模型
    _net.save()
    print('save model success')


# 计算准确率
def count_correct(_net):
    test_set = load_samples(dataset='testing_data')
    correct = 0
    for test_feature in test_set:
        predict = _net.predict(test_feature[0])
        if predict == test_feature[1][0]:
            correct += 1
    print("准确率: ", correct / len(test_set))


if __name__ == "__main__":
    # 初始化神经网络
    net = NeuralNet([28 * 28, 30, 10])

    # 训练模型
    train_model(net, True)

    # 计算准确率
    count_correct(net)

    # 预测
    result = net.predict(load_pic('test/test_0.bmp'))
    print(result)

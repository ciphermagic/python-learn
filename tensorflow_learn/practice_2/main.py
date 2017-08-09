# tensorflow 对电影评论进行分类 详细版
import tensorflow as tf
import numpy as np
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import random
import pickle

pos_file = 'pos.txt'
neg_file = 'neg.txt'


# 创建词汇表
def create_lexicon():
    result_lex = []

    # 读取文件
    def process_file(txtfile):
        with open(txtfile, 'r') as f:
            arr = []
            lines = f.readlines()
            # print(lines)
            for line in lines:
                words = word_tokenize(line.lower())
                arr += words
            return arr

    # 分词
    result_lex += process_file(pos_file)
    result_lex += process_file(neg_file)
    # print(len(result_lex))
    # 词形还原(cats->cat)
    lemmatizer = WordNetLemmatizer()
    result_lex = [lemmatizer.lemmatize(word) for word in result_lex]
    # 统计词出现次数
    word_count = Counter(result_lex)
    # print(word_count)
    # 去掉不常用的词
    result_lex = []
    for word in word_count:
        num = word_count[word]
        if 2000 > num > 20:
            result_lex.append(word)
    # print(len(result_lex))
    return result_lex


# 把每条评论转换为向量, 转换原理：
# 假设lex为['woman', 'great', 'feel', 'actually', 'looking', 'latest', 'seen', 'is'] 当然实际上要大的多
# 评论'i think this movie is great' 转换为 [0,1,0,0,0,0,0,1], 把评论中出现的字在lex中标记，出现过的标记为1，其余标记为0
# lex:词汇表；review:评论
def word_to_vector(_lex, review):
    words = word_tokenize(review.lower())
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    features = np.zeros(len(_lex))
    for word in words:
        if word in _lex:
            features[_lex.index(word)] = 1
    return features


# 将所有评论转换为向量，并拼接评论的的分类
def normalize_dataset(inner_lex):
    ds = []

    # [0,1]代表负面评论 [1,0]代表正面评论，拼接到向量后
    with open(pos_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            one_sample = [word_to_vector(inner_lex, line), [1, 0]]
            ds.append(one_sample)
    with open(neg_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            one_sample = [word_to_vector(inner_lex, line), [0, 1]]
            ds.append(one_sample)

    # print(len(ds))
    return ds


# 定义神经网络
def network(input_num, input_tensor):
    n_input = input_num  # 输入层
    n_hidden_1 = 2000  # 隐藏层 1
    n_hidden_2 = 2000  # 隐藏层 2
    n_output = 2  # 输出层

    # 定义权重和偏差值
    witght = {
        'input-h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h1-h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'h2-output': tf.Variable(tf.random_normal([n_hidden_2, n_output])),
    }
    biases = {
        'h1': tf.Variable(tf.random_normal([n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_2])),
        'output': tf.Variable(tf.random_normal([n_output])),
    }

    # 输入到隐藏层 1
    layer_1 = tf.add(tf.matmul(input_tensor, witght['input-h1']), biases['h1'])
    layer_1 = tf.nn.relu(layer_1)
    # 隐藏层 1 到隐藏层 2
    layer_2 = tf.add(tf.matmul(layer_1, witght['h1-h2']), biases['h2'])
    layer_2 = tf.nn.relu(layer_2)
    # 隐藏层 2 到输出层
    layer_output = tf.add(tf.matmul(layer_2, witght['h2-output']), biases['output'])

    return layer_output


# 整理数据
def clear_up(has_dataset):
    if not has_dataset:
        # 词典：文本中出现过的单词
        _lex = create_lexicon()
        # 词典转换为向量
        ds = normalize_dataset(_lex)
        random.shuffle(ds)
        # 把整理好的数据保存到文件，方便使用。到此完成了数据的整理工作
        with open('lex.pickle', 'wb') as f:
            pickle.dump(_lex, f)
        with open('dataset.pickle', 'wb') as f:
            pickle.dump(ds, f)
    else:
        _lex = pickle.load(open('lex.pickle', 'rb'))
        ds = pickle.load(open('dataset.pickle', 'rb'))
    return _lex, ds


# 训练模型
def train(load_data=True):
    # 加载数据
    lex, dataset = clear_up(load_data)

    # 定义变量
    input_tensor = tf.placeholder(tf.float32, [None, len(lex)], name="input")
    output_tensor = tf.placeholder(tf.float32, [None, 2], name="output")
    # 构建模型
    prediction = network(len(lex), input_tensor)
    # 计算误差
    entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=output_tensor)
    # 计算平均误差
    loss = tf.reduce_mean(entropy_loss)
    # 随机梯度下降法
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    # 初始化变量
    init = tf.global_variables_initializer()

    # 训练次数
    training_epochs = 10
    # 每次训练数量
    batch_size = 50

    # 开始训练
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(init)
        for epoch in range(training_epochs):
            epoch_loss = 0.
            mini_batches = [dataset[k:k + batch_size] for k in range(0, len(dataset), batch_size)]
            for mini_batch in mini_batches:
                mini_batch = np.array(mini_batch)
                batch_x = mini_batch[:, 0]
                batch_y = mini_batch[:, 1]
                c, _ = sess.run([loss, optimizer],
                                feed_dict={input_tensor: list(batch_x), output_tensor: list(batch_y)})
                epoch_loss += c
            print(epoch, ' : ', epoch_loss)

        # 计算准确率
        np_dataset = np.array(dataset)
        train_x = np_dataset[:, 0]
        train_y = np_dataset[:, 1]
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(output_tensor, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('准确率: ', accuracy.eval({input_tensor: list(train_x), output_tensor: list(train_y)}))

        # 保存session
        saver.save(sess, './model.ckpt')


# 使用模型预测
def predict(text):
    # 加载数据
    lex, dataset = clear_up(True)
    # 定义变量
    input_tensor = tf.placeholder(tf.float32, [None, len(lex)], name="input")
    # 构建模型
    prediction = network(len(lex), input_tensor)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, 'model.ckpt')
        features = word_to_vector(lex, text)
        res = sess.run(tf.argmax(prediction.eval(feed_dict={input_tensor: [features]}), 1))
        return res


if __name__ == "__main__":
    # train(False)
    print(predict("very good"))
    pass

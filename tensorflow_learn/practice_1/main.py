import os
import numpy as np
import tensorflow as tf
import random
import pickle
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

pos_file = 'pos.txt'
neg_file = 'neg.txt'


# 创建词汇表
def create_lexicon(p_file, n_file):
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
    result_lex += process_file(p_file)
    result_lex += process_file(n_file)
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


# 整理数据
def clear_up(has_dataset):
    if not has_dataset:
        # 词典：文本中出现过的单词
        _lex = create_lexicon(pos_file, neg_file)
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


# 定义待训练的神经网络
def neural_network(_lex, data):
    # 输入层
    n_input_layer = len(_lex)
    # hide layer(隐藏层)听着很神秘，其实就是除输入输出层外的中间层
    n_layer_1 = 2000
    n_layer_2 = 2000
    # 输出层
    n_output_layer = 2

    # 定义第一层"神经元"的权重和偏移量
    layer_1_w_b = {'w_': tf.Variable(tf.random_normal([n_input_layer, n_layer_1])),
                   'b_': tf.Variable(tf.random_normal([n_layer_1]))}
    # 定义第二层"神经元"的权重和偏移量
    layer_2_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_1, n_layer_2])),
                   'b_': tf.Variable(tf.random_normal([n_layer_2]))}
    # 定义输出层"神经元"的权重和偏移量
    layer_output_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_2, n_output_layer])),
                        'b_': tf.Variable(tf.random_normal([n_output_layer]))}

    # w*x + b
    layer_1 = tf.add(tf.matmul(data, layer_1_w_b['w_']), layer_1_w_b['b_'])
    layer_1 = tf.nn.relu(layer_1)  # 激活函数
    layer_2 = tf.add(tf.matmul(layer_1, layer_2_w_b['w_']), layer_2_w_b['b_'])
    layer_2 = tf.nn.relu(layer_2)  # 激活函数
    layer_output = tf.add(tf.matmul(layer_2, layer_output_w_b['w_']), layer_output_w_b['b_'])

    return layer_output


# 使用数据训练神经网络
def train_neural_network(has_dataset):
    _lex, _dataset = clear_up(has_dataset)
    _dataset = np.array(_dataset)
    x = tf.placeholder('float', [None, len(_dataset[0][0])])
    y = tf.placeholder('float')

    # 每次使用50条数据进行训练
    batch_size = 50

    predict = neural_network(_lex, x)
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost_func)  # learning rate 默认 0.001

    epochs = 10
    with tf.Session() as session:
        saver = tf.train.Saver()
        session.run(tf.global_variables_initializer())
        train_x = _dataset[:, 0]
        train_y = _dataset[:, 1]
        for epoch in range(epochs):
            i = 0
            epoch_loss = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = train_x[start:end]
                batch_y = train_y[start:end]
                _, c = session.run([optimizer, cost_func], feed_dict={x: list(batch_x), y: list(batch_y)})
                epoch_loss += c
                i += batch_size
            print(epoch, ' : ', epoch_loss)  #

        text_x = _dataset[:, 0]
        text_y = _dataset[:, 1]
        correct = tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('准确率: ', accuracy.eval({x: list(text_x), y: list(text_y)}))
        # 保存session
        saver.save(session, './model.ckpt')


# 使用模型预测
def prediction(text):
    _lex = pickle.load(open('lex.pickle', 'rb'))
    x = tf.placeholder('float')
    predict = neural_network(_lex, x)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(session, 'model.ckpt')
        features = word_to_vector(_lex, text)
        res = session.run(tf.argmax(predict.eval(feed_dict={x: [features]}), 1))
        return res


if __name__ == "__main__":
    # 训练模型
    train_neural_network(False)
    # 预测结果
    # print(prediction("very good"))

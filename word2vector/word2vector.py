import collections
import math
import random
import numpy as np
import tensorflow as tf
import pickle





# 生成训练样本，assert断言：申明起布尔值必须为真的判定，如果发生异常，就表示为假
def generate_batch(batch_size, window_size, data):
    # 该函数根据训练样本中词的顺序抽取形成训练集
    # 这个函数的功能是对数据data中的每个单词，分别与前一个单词和后一个单词生成一个batch
    # 即[data[1],data[0]]和[data[1],data[2]]，其中当前单词data[1]存在batch中，前后单词存在labels中
    # batch_size:每个批次训练多少样本
    # num_skips:为每个单词生成多少样本（本次实验是2个），batch_size必须是num_skips的整数倍，这样可以确保由一个目标词汇生成的样本在同一个批次中
    # window_size：单词最远可以联系的距离（本次实验设为1，即目标单词只能和相邻的两个单词生成样本），2*window_size>=num_skips
    data_index = 0
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)  # 建一个batch大小的数组，保存任意单词
    lables = np.ndarray(shape=(batch_size, 1), dtype=np.int32)  # 建一个（batch，1）大小的二维数组，保存任意单词前一个或者后一个单词，从而形成一个pair
    span = 2 * window_size + 1  # 窗口大小，为3，结构为[window_size target window_size][wn-i,wn,wn+i]
    buffer = collections.deque(maxlen=span)  # 建立一个结构为双向队列的缓冲区，大小不超过3，实际上是为了构造batch以及labels，采用队列的思想
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # batch_size一定是num_skips的倍数，保证每个batch_size都能够用完num_skips
    for i in range(batch_size // (window_size * 2)):
        target = window_size
        targets_to_avoid = [window_size]
        for j in range(window_size * 2):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * window_size * 2 + j] = buffer[window_size]
            lables[i * window_size * 2 + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, lables


def train_wordvec(vocabulary_size, batch_size, embedding_size, window_size, num_sampled, num_steps, data):
    # 定义Skip_Gram word2vector模型的网络结构
    graph = tf.Graph()
    with graph.as_default():
        # 输入数据，大小为一个batch_size=200
        train_inputs = tf.placeholder(tf.int32, shape=[None])
        # 目标数据，大小为[batch_size]
        train_labels = tf.placeholder(tf.int32, shape=[None, 1])
        # 使用cpu进行训练
        with tf.device('/cpu:0'):
            # 生成一个vocabulary_size*embedding_size的随机矩阵，为词表中的每个词，随机生成一个embedding_size维度大小的向量
            # 词向量矩阵，初始时为均匀随机正态分布，tf.random_uniform((4,4),minval=low,maxval=high,dtype=tf.float32)
            # 随机初始化一个值介于-1和1之间的随机数，矩阵大小为词表大小乘以词向量维度
            embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)
            # 全连接层，wx+b，设置w大小为，embedding_size*vocbulary_size的权重矩阵，模型内部参数矩阵，初始为截断正态分布
            nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
            # 全连接层，wx+b，设置w大小为，vocabulary_size*1的偏置
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
        # 定义loss损失函数，tf.reduce_mean求平均值，
        # 得到NCE损失（负采样得到的损失）
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,  # 权重
                                             biases=nce_biases,  # 偏差
                                             labels=train_labels,  # 输入的标签
                                             inputs=embed,  # 输入的向量
                                             num_sampled=num_sampled,  # 负采样的个数
                                             num_classes=vocabulary_size,  # 类别数目
                                             ))
        # 定义优化器，使用梯度下降优化算法
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
        # 计算每个词向量的模，并进行单位归一化，保留词向量维度
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        # 初始化模型变量
        init = tf.global_variables_initializer()

    # 基于构造网络进行训练
    with tf.Session(graph=graph) as session:
        # 初始化运行
        init.run()
        # 定义平均损失
        average_loss = 0
        # 每步进行迭代
        train_set = []
        train_set_labels = []
        for step in range(num_steps):
            for index in range(len(data)):
                batch_inputs, batch_labels = generate_batch(batch_size, window_size, data[index])
                train_set.extend(batch_inputs)
                train_set_labels.extend(batch_labels)
            # feed_dict是一个字典，在字典中需要给出每一个用到的占位符的取值
            # train_set = np.array(train_set, dtype=np.int32)
            # train_set_labels = np.array(train_set_labels, dtype=np.int32)
            feed_dict = {train_inputs: train_set, train_labels: train_set_labels}
            # 计算每次迭代中的loss
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            # 计算总loss
            average_loss += loss_val
            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                print("Average loss at step", step, ":", average_loss)
                average_loss = 0
        final_embeddings = normalized_embeddings.eval()

    return final_embeddings


# 保存embedding文件
def save_embedding(final_embeddings):
    f = open('../data/emb.txt', "w+")
    for index, item in enumerate(final_embeddings):
        f.write(str(index) + '\t' + ','.join([str(vec) for vec in item]) + '\n')
    f.close()


# 训练主函数
def train():
    # Loading data
    print('Loading data.....', flush=True)
    with open('dataset.pkl', 'rb') as f:
        data = pickle.load(f)
        user_count, item_count = pickle.load(f)
    print(len(data))
    print("data", data)
    vocabulary_size = item_count
    print(item_count)
    final_embeddings = train_wordvec(vocabulary_size, batch_size=6, embedding_size=6,
                                     window_size=1, num_sampled=3, num_steps=1, data=data)
    print(len(final_embeddings))
    save_embedding(final_embeddings)

train()

















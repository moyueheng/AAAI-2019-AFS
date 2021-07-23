import tensorflow as tf
import os

tf.app.flags.DEFINE_integer('max_step', 1000, '训练步数')
# 定义模型的路径
tf.app.flags.DEFINE_string("model_dir", './ckpt/linear_regression', "模型保存的路径+模型名字")

FLAGE = tf.app.flags.FLAGS  # 获取 max_step


def linear_regression():
    """
    自实现线性回归
    :return:None
    """
    with tf.variable_scope('initial_data'):
        # 1. 准备好数据集 y = 0.8x + 0.7 100个样本
        # 特征值 X， 目标值 y_true
        X = tf.random_normal(shape=(100, 1), mean=2, stddev=2, name='original_data_x')
        # y_true [100,1]
        # 矩阵运算 X(100, 1) * (1,1) = y_true(100, 1)
        y_true = tf.matmul(X, [[0.8]]) + 0.7

    with tf.variable_scope('linear_model'):
        # 2.建立线性模型 建立线性模型，
        # y = W * X + b, 目标： 求出权重W和偏置b
        # 随机初始化W1 b1
        weights = tf.Variable(initial_value=tf.random_normal(shape=[1, 1]),
                              name='W')  # 迁移学习  trainable=False 在迁移学习中可以制定某些网络不被学习
        # weight = tf.Variable(tf.random_normal([1, 1], mean=0.0, stddev=1.0), name=" weights", trainable=False)
        bias = tf.Variable(initial_value=tf.random_normal(shape=[1, 1]), name='b')
        y_predict = tf.matmul(X, weights) + bias  # 其实这个就是模型

    with tf.variable_scope('error'):
        # 3. 确定损失函数（预测值和真实值之间的误差） - 均方误差
        error = tf.reduce_mean(tf.square(y_predict - y_true), name='loss')  # 这是一个列向量 , error 就是loss

    with tf.variable_scope('optimizer'):
        # 4. 梯度下降优化损失： 需要指定学习率（超参数）
        # W2 = W1 - 学习率 * (方向)
        # b2 = b1 - 学习率 * (方向)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)  # optimizer 优化器

    # 收集变量
    tf.summary.scalar('error', error)  # scalar 标量， 收集0维度的张量
    tf.summary.histogram('weights', weights)  # histogram 柱状图， 收集高维度的张量
    tf.summary.histogram('bias', bias)

    # 合并
    merge = tf.summary.merge_all()

    # 初始化变量和
    init_op = tf.global_variables_initializer()
    # 创建一个saver
    saver = tf.train.Saver()

    # 开机会话
    with tf.Session() as sess:
        sess.run(init_op)
        print('随机初始化的权重为 %f, 偏置为%f ' % (weights.eval(), bias.eval()))

        # 用户board可视化 创建事件文件
        file_writer = tf.summary.FileWriter("./summary", graph=sess.graph)  # todo 后缀名有问题
        print(weights.eval(), bias.eval())

        # 加载模型
        # saver.restore(sess, './ckpt/linear_regression')
        # print(weights.eval(), bias.eval())

        # 训练
        for i in range(FLAGE.max_step):  # 学习步长
            sess.run(optimizer)
            # 添加损失， 权重，
            summary = sess.run(merge)  # op 操作存不起来， 要run才能运行
            file_writer.add_summary(summary, i)
            print('第%d步的误差为%f， 权重为%f， 偏置为%f' % (i, error.eval(), weights.eval(), bias.eval()))

            # check point： 检查点文件格式, 模型保存
            saver.save(sess, FLAGE.model_dir)


if __name__ == '__main__':
    linear_regression()

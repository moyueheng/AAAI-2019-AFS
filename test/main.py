import os

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 取消一些低版本警告

con_a = tf.constant(1.0)
con_b = tf.constant(2.0)
sum_c = tf.add(con_a, con_b)
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
print('*' * 10)

plt_a = tf.placeholder(tf.float32)
plt_b = tf.placeholder(tf.float32)
plt_add = tf.add(plt_a, plt_b)

with tf.Session(config=config) as sess:  # 设置config会打印设备信息
    # res_c = sess.run(sum_c)  # 其实就是取出这个OP的值
    # print('res_c:', res_c)
    # print('con_a:', con_a)
    # print('con_b:', con_b)
    # print('sum_c:', sum_c)
    file_writer = tf.summary.FileWriter("./summary", graph=sess.graph)

    res = sess.run(plt_add, feed_dict={plt_a: 1.0, plt_b: 2.0})
    print(con_a.eval())
    # print(res.eval())
    print(res)

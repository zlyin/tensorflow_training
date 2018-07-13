# import
import tensorflow as tf
import numpy as np

# restore saved netwrok
# 提取时, 先建立零时的W 和 b容器. 找到文件目录,
# 并用saver.restore()我们放在这个目录的变量.
W_res = tf.Variable( np.arange(6).reshape((2,3)), dtype = tf.float32, name =
        'weights')
b_res = tf.Variable( np.arange(3).reshape((1,3)), dtype = tf.float32, name =
        'biases')
# reload it from saver.restore
saver2 = tf.train.Saver()
with tf.Session() as sess2:
    saver2.restore(sess2, "my_net/save_net.ckpt")
    print("weights resotred = ", sess2.run(W_res))
    print("bias restored = ", sess2.run(b_res))




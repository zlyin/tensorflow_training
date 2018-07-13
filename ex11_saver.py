# 建好了一个神经网络, 训练好了, 肯定也想保存起来, 用于再次加载.
# use save module

import tensorflow as tf
import numpy as np

# save to file
# Important = define the same dtype & shape when restore
W = tf.Variable([ [1,2,3], [3,4,5] ], dtype = tf.float32, name= 'weights')
b = tf.Variable([ [1,2,3] ], dtype = tf.float32, name = 'biases')
# initiate all variables
init = tf.global_variables_initializer()

# create session & saver
# 首先要建立一个 tf.train.Saver() 用来保存, 提取变量.
# 再创建一个名为my_net的文件夹, 用这个 saver 来保存变量到这个目录
# "my_net/save_net.ckpt".

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess, "my_net/save_net.ckpt")    # ckpt = check point
    print("save to path: ", save_path)



# restore saved netwrok
# 提取时, 先建立零时的W 和 b容器. 找到文件目录,
# 并用saver.restore()我们放在这个目录的变量.
# W_res = tf.Variable( np.arange(6).reshape((2,3)), dtype = tf.float32, name =
#         'weights')
# b_res = tf.Variable( np.arange(3).reshape((1,3)), dtype = tf.float32, name =
#         'biases')
# # reload it from saver.restore
# saver2 = tf.train.Saver()
# with tf.Session() as sess2:
#     saver2.restore(sess2, "my_net/save_net.ckpt")
#     print("weights resotred = ", sess2.run(W_res))
#     print("bias restored = ", sess2.run(b_res))
# 
# 


# ensorflow 中的 placeholder , placeholder 是 Tensorflow
# 中的占位符，暂时储存变量.
# Tensorflow 如果想要从外部传入data, 那就需要用到 tf.placeholder(),
# 然后以这种形式传输数据 sess.run(***, feed_dict={input: **}).


# import 
import tensorflow as tf

#define a tf placeholder with dtype = float32
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

# mul = multiply
output = tf.multiply(input1, input2)

# 传值的工作交给了 sess.run() , 需要传入的值放在了feed_dict={} 并一一对应每一个
# input. placeholder 与 feed_dict={} 是绑定在一起出现的。
with tf.Session() as sess:
    print(sess.run(output, feed_dict ={input1:[7.0], input2:[2.0]})) #  [[14.0]]

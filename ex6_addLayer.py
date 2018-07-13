# Tensorflow里定义一个添加层的函数可以很容易的添加神经层,为之后的添加省下不少时间.
# 神经层里常见的参数通常有weights、biases和激励函数。

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Step1.  define layer
def add_layer(input, in_size, out_size, activation_function=None):
    # initialize weights
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # initialize biases, better not be 0
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    # define Wx_plus_b 即神经网络未激活的值
    Wx_plus_b = tf.matmul(input,weights)+biases

# 当activation_function——激励函数为None时，
# 输出就是当前的预测值——Wx_plus_b，不为None时，
# 就把Wx_plus_b传到activation_function()函数中得到输出。
    if activation_function is None:
        output = Wx_plus_b
    else:
        output = activation_function(Wx_plus_b)
    return output

#Step2.   input data
# x_data和y_data并不是严格的一元二次函数的关系，
# 因为我们多加了一个noise,这样看起来会更像真实情况
x_data = np.linspace(-1, 1,300, dtype = np.float32)[:,np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) -0.5*noise
# placeholder to hold input
xs = tf.placeholder(tf.float32, [None,1])   # None = any inputs
ys = tf.placeholder(tf.float32, [None,1])

# define network 我们构建的是——输入层1个、隐藏层10个、输出层1个的神经网络。
# Tensorflow 自带的激励函数tf.nn.relu
l1 = add_layer(xs, 1, 10, activation_function = tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_function=None)
# cal loss func
loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))

# 很关键的一步，如何让机器学习提升它的准确率
# tf.train.GradientDescentOptimizer()中的值通常都小于1
# 这里取的是0.1，代表以0.1的效率来最小化误差loss。
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# initiate all variables
init = tf.global_variables_initializer()

# use session to do init
sess = tf.Session()
sess.run(init)

# do training
# 让机器学习1000次。机器学习的内容是train_step, 用 Session 来 run 每一次
# training 的数据，逐步提升神经网络的预测准确性

#for i in range(1000):
#    sess.run(train_step,feed_dict={xs:x_data, ys:y_data})
#    if i % 50 == 0:
#        print("loss = ", sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
#

#Step3.  Visualization with matplotlib
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

# plot both scatter plot & regression line
for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data, ys:y_data})
    if i % 50 == 0:
        # visualize the result & improvement
        try:
            ax.lines.remove(line[0])
        except Exception:
            pass
        prediction_val = sess.run(prediction, feed_dict={xs:x_data})
        # plot prediction val
        lines = ax.plot(x_data, prediction_val, 'r-', lw=1)
        plt.savefig("ex6_trainloss&prediction.png") 
        print("loss = ", sess.run(loss, feed_dict={xs:x_data, ys:y_data}))



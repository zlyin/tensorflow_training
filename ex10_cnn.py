# import packages
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# define compute_accuracy
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pred = sess.run(prediction, feed_dict={xs:v_xs, ys:v_ys, keep_prob :1})
    correct_prediction = tf.equal( tf.argmax(y_pred, 1), tf.argmax(v_ys, 1) )
    accuracy = tf.reduce_mean( tf.cast( correct_prediction, tf.float32) )
    result = sess.run( accuracy, feed_dict={ xs:v_xs, ys:v_ys, keep_prob: 1} )
    return result

# define weights
def weight_var(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

# define biases
def bias_var(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

# define convo layer
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')
# tf.nn.conv2d函; x = pic parameters, W = layer weights;
# 定义步长strides=[1,1,1,1]值，strides[0]和strides[3]的两个1是默认值，
# 中间两个1代表padding时在x方向运动一步，y方向运动一步，
# padding采用的方式是SAME -> 输出图片的大小没有变化依然是28x28

# define pooling
# 用pooling来稀疏化参数，也就是卷积神经网络中所谓的下采样层。
# pooling = 最大值池化 or 平均值池化，本例采用的是最大值池化tf.max_pool()。
# 池化的核函数大小为2x2，因此ksize=[1,2,2,1]，步长为2，因此strides=[1,2,2,1]:
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides=[1,2,2,1], padding =
    'SAME')


# pic processing
xs = tf.placeholder(tf.float32, [None, 28*28])  # 28*28 = 784
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

# convert xs -> shape = [-1, 28, 28, 3/1]
# -1代表先不考虑输入的图片例子多少这个维度，后面的1是channel的数量，
# 输入的图片是黑白的，因此channel是1，例如如果是RGB图像，那么channel就是3。
x_image = tf.reshape(xs, [-1, 28, 28,1])

# create conv layer
# 本层我们的卷积核patch的大小是5x5，因为黑白图片channel是1所以输入是1，
# 输出是32个featuremap
W_conv1 = weight_var([5,5,1,32])
b_conv1 = bias_var([32])
# 1st conv layer
h_conv1 = tf.nn.relu( conv2d(x_image, W_conv1) + b_conv1)
# 1st pooling
h_pool1 = max_pool_2x2(h_conv1)  # output = 14*14*32

# 2nd conv layer
W_conv2 = weight_var([5,5,32,64])   # input channel =32, output channel = 64
b_conv2 = bias_var([64])
# 2nd conv layer
h_conv2 = tf.nn.relu( conv2d(h_pool1, W_conv2) + b_conv2)    # output = 14*14*64
# 2nd polling
h_pool2 = max_pool_2x2(h_conv2) # output = 7*7*64

# create FC layer
# 通过tf.reshape()将h_pool2的输出值从一个三维的变为一维的数据,
# -1表示先不考虑输入图片例子维度, 将上一个输出结果展平
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# 此时weight_variable的shape输入就是第二个卷积层展平了的输出大小:
# 7x7x64， 后面的输出size我们继续扩大，定为1024

# 1st FC layer
W_fc1 = weight_var([7*7*64, 1024])
b_fc1 = bias_var([1024])
# multiply h_pool2_falt & W_fc1
h_fc1 = tf.nn.relu( tf.matmul(h_pool2_flat, W_fc1) + b_fc1 )
# 考虑过拟合问题，可以加一个dropout的处理
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 2nd FC layer
W_fc2 = weight_var([1024, 10])
b_fc2 = bias_var([10])

# 用softmax分类器（多分类，输出是各个类的概率）,对我们的输出进行分类
prediction = tf.nn.softmax( tf.matmul(h_fc1_drop, W_fc2) +  b_fc2)

# define loss func
cross_entropy = tf.reduce_mean( -tf.reduce_sum(ys * tf.log(prediction),
    reduction_indices=[1]))

# define Adam as optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# define session & traing process
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict = { xs:batch_xs, ys:batch_ys, keep_prob:0.5})
    if i % 50 == 0:
        print("epoch %d, accuracy = %f" %(i, compute_accuracy(mnist.test.images[:1000],
            mnist.test.labels[:1000])))


















# import tf & dataset
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# 数据中包含55000张训练图片，每张图片的分辨率是28×28，
# 所以我们的训练网络输入应该是28×28=784个像素数据

#define add_layer function
def add_layer(inputs, in_size, out_size, activation_function = None):
    # add 1 more layer
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, weights) + biases
    # if there is activation function
    if activation_function is None:
        output = Wx_plus_b
    else:
        output = activation_function(Wx_plus_b)
    return output

# define compute_accuracy function
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_predict = sess.run(prediction, feed_dict = {xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys})
    return result

# construct network
xs = tf.placeholder(tf.float32, [None, 28*28])
ys = tf.placeholder(tf.float32, [None,10])

# add_layer
prediction = add_layer(xs, 28*28, 10, activation_function=tf.nn.softmax)

# cross-entropy loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
    reduction_indices = [1]))

# train_step & optimizer
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# begin training
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict = {xs:batch_xs, ys:batch_ys})
    if i %50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))





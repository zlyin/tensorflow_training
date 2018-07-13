# import 
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

# define add_layer()
def add_layer(inputs, in_size, out_size, layer_name, activation_func = None):
    # add more layer
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs, weights) + biases # matmul order can't be reversed!
    # here to drop out
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_func is None:
        output = Wx_plus_b
    else:
        output = activation_func(Wx_plus_b)
    tf.summary.histogram(layer_name + '/outputs', output)
    return output

# 这里的keep_prob是保留概率，即我们要保留的结果所占比例，它作为一个placeholder，在run时传入，
# 当keep_prob=1的时候，相当于100%保留，也就是dropout没有起作用。 
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None,64])
ys = tf.placeholder(tf.float32, [None,10])

# prepare data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# add layers
l1 = add_layer(xs, 64, 50, 'l1', activation_func = tf.nn.tanh)
prediction = add_layer(l1, 50, 10, 'l2', activation_func= tf.nn.softmax)

# define loss func
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
    reduction_indices = [1]))
tf.summary.scalar('cross_entropy', cross_entropy)

# defeine optimizer
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


##  define training
sess = tf.Session()
merged = tf.summary.merge_all()
# summary writer
train_writer = tf.summary.FileWriter("logs/train", sess.graph)
test_writer = tf.summary.FileWriter("logs/test", sess.graph)
# initialize
sess.run(tf.global_variables_initializer())
# training epochs
for i in range(1000):
    # choose drop out
    sess.run(train_step, feed_dict = {xs: X_train, ys:y_train, keep_prob:0.5 })
    if i % 50 == 0:
        # record loss
        train_result = sess.run(merged, feed_dict = { xs:X_train, ys:y_train,
            keep_prob : 1})
        test_result = sess.run(merged, feed_dict = { xs:X_test, ys:y_test,
            keep_prob : 1})
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)
        print("epoch %d, loss = %f" %(i,sess.run(cross_entropy,
            feed_dict={xs:X_train, ys:y_train, keep_prob : 1.0})))



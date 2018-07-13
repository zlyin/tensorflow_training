import tensorflow as tf
import numpy as np

# define a placeholder
# xs = tf.placeholder(tf.float32, [None,1])
# ys = tf.placeholder(tf.float32, [None,1])

# 对于input我们进行如下修改： 首先，可以为xs指定名称为x_in:
# xs = tf.placeholder(tf.float32, [None,1], name='x_in')
# ys = tf.placeholder(tf.float32, [None,1], name='y_in')

# 使用with tf.name_scope('inputs')可以将xs和ys包含进来，形成一个大的图层，图层的名字就是with
# tf.name_scope()方法里的参数。
with tf.name_scope('inputs'):
    # define placeholder
    xs = tf.placeholder(tf.float32, [None,1])
    ys = tf.placeholder(tf.float32, [None,1])

# define add_layer method with name_scope way:
def add_layer(inputs, in_size, out_size, n_layer, activation_function = None):
    # add oen more layer and return output
    layer_name = 'layer%s'%n_layer  # define a new layer with name
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
            # add a histogram_summary for weights
            tf.summary.histogram(layer_name + '/weights', weights)
        
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size])+0.1, name='b')
            # add a histogram_summary for biases
            tf.summary.histogram(layer_name + '/biases', biases)
        
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, weights), biases)
        
        if activation_function is None:
            output = Wx_plus_b
        else:
            output = activation_function(Wx_plus_b)
       
        # add a histogram_summary for output
        tf.summary.histogram(layer_name + '/output', output)
        return output

# add hidden layer
l1 = add_layer(xs, 1, 10, n_layer = 1, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, n_layer = 2,  activation_function = None)

# error btw prediction & real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
        reduction_indices = [1]))
    # for loss  使用的是tf.summary.scalar() 方法
    tf.summary.scalar('loss', loss)

# train_step
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# define a graph
sess = tf.Session()
# merge all summary
merged = tf.summary.merge_all()
# get a fileWriter
writer = tf.summary.FileWriter("logs/", sess.graph)
# initialize all variables
init = tf.global_variables_initializer()
sess.run(init)


## make up some data
x_data = np.linspace(-1,1,300, dtype = np.float32)[:,np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise


# define traing 1000 epochs
for i in range(1000):
    sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
    # 较为直观显示训练过程中每个参数的变化，我们每隔上50次就记录一次结果 ,
    # 同时我们也应注意, merged 也是需要run 才能发挥作用的
    if i%50 == 0:
        rs = sess.run(merged, feed_dict={xs:x_data, ys:y_data})
        writer.add_summary(rs, i)   #add summary of ith epoch
        # print out loss 
        print("loss at epoch %d, %f" %(i, sess.run(loss, feed_dict={xs:x_data,
            ys:y_data}))) 






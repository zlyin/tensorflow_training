# import tensorflow
import tensorflow as tf
import numpy as np

print("import successfully")

# create date 
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1+0.3
print("x = ",x_data)
print("y = " , y_data)


# create learning model
Weights = tf.Variable(tf.random_uniform([1], -1.0,1.0))
biases = tf.Variable(tf.zeros([1]))
y = Weights*x_data + biases

# loss func
loss = tf.reduce_mean(tf.square(y-y_data))

# back propagate
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# training
# 只是建立了神经网络的结构, 还没有使用这个结构. 在使用这个结构之前,
# 我们必须先初始化所有之前定义的Variable, 所以这一步是很重要的!
init = tf.global_variables_initializer()

# initiate sessions
session = tf.Session()
session.run(init)   # Very important!

for step in range(201):
    session.run(train)
    if step % 20 == 0:
        print("step %d, weights = %f, biases = %f" %(step, session.run(Weights),
            session.run(biases)))


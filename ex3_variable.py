# 在 Tensorflow 中，定义了某字符串是变量，它才是变量，这一点是与 Python 所不同的

import tensorflow as tf

# create varaible
state = tf.Variable(0, name='counter')  # dtype=int32_ref

# define const
one = tf.constant(1)

# define add, but it doesn't do addint operation yet
new_val = tf.add(state, one)

# state = new_val
update = tf.assign(state, new_val)
print("state = ", state, "& new_val = ", new_val)

# 在 Tensorflow 中设定了变量，那么初始化变量是最重要的！！所以定义了变量以后,
# 一定要定义init = tf.global_variables_initializer()  
init = tf.global_variables_initializer()
with tf.Session() as sess1:
    sess1.run(init)
    for _ in range(3):
        sess1.run(update)
        print(sess1.run(state))
        print("just print(state) = ", state)    
        # doesn't work at all <tf.Variable 'counter:0' shape=() dtype=int32_ref>
        # 一定要把 sess 的指针指向 state 再进行 print 才能得到想要的结果！





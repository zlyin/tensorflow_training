# Tensorflow 中的 Session, Session 是 Tensorflow
# 为了控制,和输出文件的执行的语句. 运行 session.run()
# 可以获得你要得知的运算结果, 或者是你所要运算的部分.

# import tf
import tensorflow as tf

# create 2 matrix
matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])
product = tf.matmul(matrix1, matrix2)

# 因为 product 不是直接计算的步骤, 所以我们会要使用 Session 来激活 product
# 并得到计算结果
# method1
sess1 = tf.Session()
result1 = sess1.run(product)
print("result1 = ", result1)        # [[12]]
sess1.close()

# method2
with tf.Session() as sess2:
    result2 = sess2.run(product)
    print("result2 = ", result2)    #[[12]]






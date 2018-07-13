# 激励函数来处理自己的问题, 不过要确保的是这些激励函数必须是可以微分的, 因为在
# backpropagation 误反向传递的时候,
# 只有这些可微分的激励函数才能把误差传递回去.

# 在卷积神经网络 Convolutional neural networks 的卷积层中, 推荐的激励函数是
# relu. 在循环神经网络中 recurrent neural networks, 推荐的是 tanh 或者是 relu 

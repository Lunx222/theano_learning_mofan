import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import layer


# Make up some fake data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]  # newaxis用于在现有数组中增加一个新的维度
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + noise  # y = x^2 - 0.5

# show the fake data
# plt.scatter(x_data, y_data)
# plt.show()

# determine the inputs dtype
# dtype 一定要一致
x = T.dmatrix('x')
y = T.dmatrix('y')


# add layers
l1 = layer.Layer(x, 1, 10, T.nnet.relu) # 隐藏层
l2 = layer.Layer(l1.outputs, 10, 1, None)  # 输出层

# compute the cost
cost = T.mean(T.square(l2.outputs - y)) # 求300个样本的平均误差

# compute the gradients
gW1, gb1, gW2, gb2 = T.grad(cost,[l1.W, l1.b, l2.W, l2.b])  # cost 越大 梯度下降幅度越大  求导

# apply gradient descent
learning_rate = 0.05  # 在NN中一般小于1，大于1是完整的
train = theano.function(
    inputs=[x,y],
    outputs = cost,
    updates=[(l1.W, l1.W - learning_rate * gW1),
             (l1.b, l1.b - learning_rate * gb1),
             (l2.W, l2.W - learning_rate * gW2),
             (l2.b, l2.b - learning_rate * gb2)]

)

# prediction
predict = theano.function(inputs=[x], outputs=l2.outputs)

# plot the fake data
fig = plt.figure()
ax = fig.add_subplot(1,1,1) # 第一行第一列的第一个图片
ax.scatter(x_data, y_data)  # 散点效果
plt.ion()  # 打开交互模式，可以不阻塞地绘图
plt.show()

for i in range(1000):
    # training
    err = train(x_data, y_data)
    if i % 50 == 0:
        # print(err)
        # to visualize the result and improvement
        # 忽略第一次执行时没有线
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        predicted_value = predict(x_data)
        # plot the prediction
        lines = ax.plot(x_data, predicted_value, 'r-', lw = 5)  # lw为线地宽度
        plt.pause(1)

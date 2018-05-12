import theano
import theano.tensor as T
import numpy as np


"""
to define the layer like this:
l1 = Layer(input, in_size=1, out_size=10, activation_function)
l2 = Layer(l1.outputs, 10, 1, None)  # None 是linear function
"""

class Layer(object):
    def __init__(self,inputs ,in_size, out_size, activatioin_function=None):
        self.W = theano.shared(np.random.normal(0,1,(in_size,out_size)))   # 乱序利于神经网络的学习
        self.b = theano.shared(np.zeros((out_size, )) + 0.1)   # 使bias保持一个较小的正数，便于更新
        self.Wx_plus_b = T.dot(inputs, self.W) + self.b
        self.activation_function = activatioin_function
        if activatioin_function is None:
            self.outputs = self.Wx_plus_b
        else:
            self.outputs = self.activation_function(self.Wx_plus_b)  # 激励一下



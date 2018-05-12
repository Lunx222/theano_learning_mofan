# 使用GPU并行，或是multiprocessing时，需要用到共享变量
import numpy as np
import theano
import theano.tensor as T


state = theano.shared(np.array(0,dtype=np.float64),'state')
inc = T.scalar('inc', dtype=state.dtype)  # 要统一变量类型，因此从现有的变量中调出其类型
accumulator = theano.function([inc], state, updates=[(state, state + inc)])

# to get variable value
print(state.get_value())
accumulator(1)
print(state.get_value())
accumulator(10)
print(state.get_value())

# to set variable value
# 用于保存和提取model参数
state.set_value(-1)
accumulator(3)
print(state.get_value())

# temporarily replace shared variable with another value in another function
tmp_func = state*2 + inc
a = T.scalar(dtype=state.dtype)  # 用a代替state
skip_shared = theano.function([inc,a], tmp_func, givens=[(state,a)])  # givens表示用a代替state
print(skip_shared(2,3))
print(state.get_value())
# 结果，原来的state不会被改变

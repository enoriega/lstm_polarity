import dynet as dy
import numpy as np


A = dy.inputTensor([1,2,3,4])
B = dy.inputTensor([5,6,7,8])
C = dy.inputTensor([9,10,11,12])

D = [A,B,C]
E = dy.concatenate(D, d=1)

F_mat = dy.kmax_pooling(x=E, k=1, d=1)

print(E.dim())

print(F_mat.npvalue())


params = dy.ParameterCollection()
builder = dy.LSTMBuilder(1, 4, 4, params)


lstm = builder.initial_state()
outputs = lstm.transduce(D)

print(outputs)

F_reshape = dy.reshape(F_mat, d = (4,))

print(F_reshape.dim())
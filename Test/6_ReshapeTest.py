import dynet as dy


A = dy.inputTensor([[1,2,3,4]])

print(A.dim())

B = dy.reshape(A, (4,))

print(B.dim())

import dynet as dy

mat1 = dy.inputTensor([[1,2], [3,4]]) # Row major
mat2 = 1+mat1
print(mat2.npvalue())

import numpy as np
import dynet as dy
np.random.seed(2)

# dynet feature: when the input is a vector, it is automatically represented as column vector by default.

A = dy.inputTensor([1,2,3,4])
B = dy.inputTensor([5,6,7,8])
C = dy.inputTensor([9,10,11,12])
D = dy.inputTensor([13,14,15,16])
E = dy.inputTensor([17,18,19,20])


input_list = [A,B,C,D,E]

input_tensor = dy.transpose(dy.concatenate(input_list,d=1))



#print(input_tensor.dim())
#print(input_tensor[0].npvalue())
#print(input_tensor[3].npvalue())

((n_row, n_col), batch_dim) = input_tensor.dim()


input_filter = dy.inputTensor([[0.5], [0.1], [0.2]])
output=list([])
for i in np.arange(n_row-2)+1:
  print('left bound %d, right bound %d' % (i-1, i+2))
  z_i = input_tensor[i-1:i+2]
  output.append(dy.filter1d_narrow(z_i, input_filter))

output_tensor = dy.concatenate(output, d=0)
  
print('output tensor:',output_tensor.npvalue())

output_vec = dy.kmax_pooling(x=output_tensor, k=1, d=0)
print('output vector:',output_vec.npvalue())
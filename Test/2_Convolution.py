import numpy as np
import dynet as dy
np.random.seed(2)

X = dy.inputTensor(np.random.rand(2,4,4,1))
f = dy.inputTensor(np.array([[[1,1],[1,1]],[[1,1],[1,1]]]))
print(X.npvalue())

Y = dy.conv2d(x=X, f=f, stride=[2,2])
print(Y.npvalue())

# here are some examples of conv operation in dynet:
# https://github.com/neubig/nn4nlp-code/blob/master/05-cnn/cnn-class.py#L48


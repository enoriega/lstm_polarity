import dynet as dy

vec1 = dy.inputTensor([1,2,3,4])
vec2 = dy.inputTensor([5,6,7,8])

vec3 = dy.concatenate([vec1, vec2])
print(vec3.npvalue())

print(vec1.dim())
print(vec2.dim())
print(vec3.dim())

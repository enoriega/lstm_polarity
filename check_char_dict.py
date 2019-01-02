import pickle

with open('char_dict.pickle', 'rb') as handle:
    char_embeddings = pickle.load(handle)
    
    
print(char_embeddings)

char_embeddings['']=1

print('' in char_embeddings)
print(' ' in char_embeddings)


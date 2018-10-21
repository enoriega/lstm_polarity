import numpy as np
import pickle




medPubEmbd = 'embeddings_november_2016.txt'


def loadMedPubModel(medPubEmbd):
    print("Loading Glove Model")
    f = open(medPubEmbd,'r', encoding='utf-8')
    model = {}
    for i, line in enumerate(f):
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
        if i%10000==0:
            print('Processing line ',i,' ...')
    print("Done.",len(model)," words loaded!")
    return model
    
def save_obj(dictObj, filename ):
    with open(filename + '.pkl', 'wb') as f:
        pickle.dump(dictObj, f, pickle.HIGHEST_PROTOCOL)
    
medPubDict = loadMedPubModel(medPubEmbd)
save_obj(medPubDict,'medPubDict')



#def load_obj(name ):
#    with open('obj/' + name + '.pkl', 'rb') as f:
#        return pickle.load(f)

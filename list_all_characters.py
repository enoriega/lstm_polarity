import pandas as pd
import numpy as np
import time
from string import ascii_lowercase
import pickle

sentences = pd.read_csv("SentencesInfo_all_label.csv", encoding = "ISO-8859-1")

char_dict = {}

start_time = time.time()
for row in np.arange(len(sentences)):
  sentence = sentences.iloc[row]['sentence text'].lower()
  for character in sentence:
    if character not in char_dict:
      char_dict[character]=1
    else:
      char_dict[character]+=1
  if row%2000==0:
    print('processing sentence ',row+1,' ...')

print('dict:', char_dict)
print('length of dict:', len(char_dict))
print('time consumption:',time.time()-start_time, 's')

start_char = 'a'
i=0

for i in ascii_lowercase:
  print(i,' is in dict:', i in char_dict)
  
with open('char_dict.pickle', 'wb') as handle:
  pickle.dump(char_dict, handle)

# coding: utf-8

# In[3]:


import numpy as np
import bcolz
import pickle
#import pandas as pd
glove_path = "Path to target folder for embedding"
words = []
idx = 0
word2idx = {}
vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.300.dat', mode='w')

with open(f"Path to downloaded embedding file(glove.6B.300d.txt)", 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.array(line[1:]).astype(np.float)
        vectors.append(vect)
        if idx%10000 == 0: print(idx)

vectors = bcolz.carray(vectors[1:].reshape((400000, 300)), rootdir=f'{glove_path}/6B.300.dat', mode='w')
vectors.flush()
pickle.dump(words, open(f'{glove_path}/6B.300_words.pkl', 'wb'))
pickle.dump(word2idx, open(f'{glove_path}/6B.300_idx.pkl', 'wb'))


# In[4]:


vectors = bcolz.open(f'{glove_path}/6B.300.dat')[:]
words = pickle.load(open(f'{glove_path}/6B.300_words.pkl', 'rb'))
word2idx = pickle.load(open(f'{glove_path}/6B.300_idx.pkl', 'rb'))

glove = {w: vectors[word2idx[w]] for w in words}


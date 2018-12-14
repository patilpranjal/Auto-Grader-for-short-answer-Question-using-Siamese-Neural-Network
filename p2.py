import numpy as np
import bcolz
import pickle
import pandas as pd
import string
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer, PorterStemmer

def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    words = message.split(" ")
    word = [x.lower() for x in words]
    for i in range(len(word)):
        table = str.maketrans({key: None for key in string.punctuation})
        word[i] = word[i].translate(table)

    return word

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = PorterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def findfirstinst(stemmed,original):
    for i in range(len(words)):
        if stemmed == stem_words(words[i]):
            a = words[i]
            break
        else:
            if original == words[i]:
                a =words[i]
                break
    return a


glove_path = "E:/Acads - Stanford/CS229/Project"
vectors = bcolz.open(f'{glove_path}/6B.50.dat')[:]
words = pickle.load(open(f'{glove_path}/6B.50_words.pkl', 'rb'))
word2idx = pickle.load(open(f'{glove_path}/6B.50_idx.pkl', 'rb'))
#words_mod = {stem_words(w):w for w in words}

glove = {w: vectors[word2idx[w]] for w in words}

df = pd.read_csv("E:\Acads - Stanford\CS229\Project\Book2.csv")

W1 = df["studentAnswer"]
W_original = []
W =[]
for i in range(len(W1)):
    W_original.append(get_words(W1[i]))
    W.append(stem_words((get_words(W1[i]))))

s1 = df["accuracy"]
s = np.zeros(len(s1))
for i in range(len(s1)):
    if s1[i] == "correct":
        s[i] = 1

print(W[1][9])

t = []
for i in range(len(W)):
    temp = np.zeros(50)
    for j in range(len(W[i])):
        temp = (1/len(W[i]))*glove[str(findfirstinst(W[i][j], W_original[i][j]))] + temp
    t.append(temp)
y_pred = np.zeros(len(W))



for i in range(len(W)):
    f = np.zeros(len(W))
    for j in range(len(W)):
        f[j] = np.dot(t[i],t[j])/np.sqrt(np.dot(t[i],t[i])*np.dot(t[j],t[j]))
    args = np.argsort(f)
    if s[args[30]] + s[args[34]] + s[args[33]] + s[args[32]] + s[args[31]] > 0:
        y_pred[i] = 1

loss = 0
for i in range(len(y_pred)):
    if y_pred[i] != s[i]:
        loss = loss + 1
loss = loss/len(y_pred)
print(loss)





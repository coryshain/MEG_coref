import sys
import os
import pickle
import random
random.seed(11112020)
import numpy as np
np.random.seed(11112020)
import pandas as pd
from nltk.corpus import brown
from nltk.probability import ConditionalFreqDist, ConditionalProbDist, ELEProbDist

sys.stderr.write('Computing Brown corpus statistics...\n')
brown_wrds = [(x[0], x[1].split('-')[0]) for x in brown.tagged_words()]
cfdist = ConditionalFreqDist(brown_wrds)
cpdist = ConditionalProbDist(cfdist, ELEProbDist, 12)

sys.stderr.write('Reading frequency tables...\n')
df = pd.read_csv('stim_src/SUBTLEXusfrequencyabove1.csv')
frequencies = dict(zip(df.Word, df.FREQcount))

embeddings = {}
sys.stderr.write('Reading word embeddings...\n')
sys.stderr.flush()
if os.path.exists('stim_src/glove.obj'):
    with open('stim_src/glove.obj', 'rb') as f:
        embeddings = pickle.load(f)
else:
    with open('stim_src/glove.840B.300d.txt', 'rb') as f:
        for i, l in enumerate(f):
            if i % 1000 == 0:
                sys.stderr.write('\r%d lines processed...' % i)
            s = l.split()
            embeddings[str(s[0], 'utf-8')] = np.array([float(s[i]) for i in range(1, len(s))])
    with open('stim_src/glove.obj', 'wb') as f:
        pickle.dump(embeddings, f)

sys.stderr.write('\nEmbeddings loaded.\n')




# OCCUPATIONS

words = []
wordset = set()
freqs = []
feats = []
pred_types = []
with open('stim_src/occupations_1w.txt', 'r') as f:
    for l in f:
        w = l.strip()
        if w in embeddings and w in frequencies:
            words.append(w)
            freqs.append(frequencies[w])
            feats.append(embeddings[w])
            pred_types.append('is-a')
            wordset.add(w)

with open('stim_src/evs_occupations.txt', 'r') as f:
    for l in f:
        w = l.strip()
        if not w in wordset and w in embeddings and w in frequencies:
            words.append(w)
            freqs.append(frequencies[w])
            feats.append(embeddings[w])
            pred_types.append('is-a')
            wordset.add(w)


# NOUNS

nouns = pd.read_csv('stim_gen/has_a_trigram.csv')
# nouns = nouns[nouns.prefix.isin(['have a', 'has a', 'had a', 'own a', 'owns a'])]
# nouns = nouns[nouns.prefix.isin(['own a', 'owns a'])]
# thresh = np.quantile(nouns.logprob, 0.5)
# nouns = nouns[nouns.logprob > thresh]
nouns = set(nouns.word)

for w in nouns:
    if not w in wordset and w in embeddings and w in frequencies and w in cpdist and cpdist[w].max() == 'NN':
        words.append(w)
        freqs.append(frequencies[w])
        feats.append(embeddings[w])
        pred_types.append('owns-a')
        wordset.add(w)


# VERBS

verbs = pd.read_csv('stim_gen/likes_to_trigram.csv')
# verbs = verbs[verbs.prefix.isin(['like to', 'likes to', 'liked to'])]
# thresh = np.quantile(verbs.logprob, 0.5)
# verbs = verbs[verbs.logprob > thresh]
verbs = set(verbs.word)

for w in verbs:
    if not w in wordset and w in embeddings and w in frequencies and w in cpdist and cpdist[w].max() in ['VB', 'VBP']:
        words.append(w)
        freqs.append(frequencies[w])
        feats.append(embeddings[w])
        pred_types.append('likes-to')
        wordset.add(w)

if not os.path.exists('stim_gen'):
    os.makedirs('stim_gen')

with open('stim_gen/all_preds.txt', 'w') as f:
    for i, (w, p, r, d) in enumerate(zip(words, pred_types, freqs, feats)):
        f.write('%s %s %s %s\n' % (w, p, r, ' '.join([str(x) for x in d])))
    f.write('\n')

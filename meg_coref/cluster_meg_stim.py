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
from nltk.util import ngrams
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


def ethnicity_map(s):
    if s == 'ASIAN AND PACI':
        out = 'ASIAN AND PACIFIC ISLANDER'
    elif s == 'BLACK NON HISP':
        out = 'BLACK NON HISPANIC'
    elif s == 'WHITE NON HISP':
        out = 'WHITE NON HISPANIC'
    else:
        out = s

    return out


# NAMES

names_df = pd.read_csv('data/MEG/nyc_names_12-23-20.csv')
names_df = names_df.sort_values(['Ethnicity', 'Gender', 'Count'], ascending=[1, 1, 0])
names_df.Ethnicity = names_df.Ethnicity.map(ethnicity_map)
names_df["Child's First Name"] = names_df["Child's First Name"].str.capitalize()
names_df = names_df.drop_duplicates(['Ethnicity', 'Gender', "Child's First Name"])
N = 30
with open('output/MEG/stim/names.txt', 'w') as f:
    for key, df in names_df.groupby(['Ethnicity', 'Gender']):
        f.write('%s, %s:\n' % key)
        for index, row in df.head(N).iterrows():
            f.write('%s\n' % row["Child's First Name"].capitalize())
        f.write('\n')


sys.stderr.write('Computing Brown corpus statistics...\n')
brown_wrds = [(x[0], x[1].split('-')[0]) for x in brown.tagged_words()]
cfdist = ConditionalFreqDist(brown_wrds)
cpdist = ConditionalProbDist(cfdist, ELEProbDist, 12)

ALGO = KMeans
K = 50

sys.stderr.write('Reading frequency tables...\n')
df = pd.read_csv('data/MEG/SUBTLEXusfrequencyabove1.csv')
frequencies = dict(zip(df.Word, df.FREQcount))

embeddings = {}
sys.stderr.write('Reading word embeddings...\n')
sys.stderr.flush()
if os.path.exists('data/MEG/glove.obj'):
    with open('data/MEG/glove.obj', 'rb') as f:
        embeddings = pickle.load(f)
else:
    with open('data/glove.840B.300d.txt', 'rb') as f:
        for i, l in enumerate(f):
            if i % 1000 == 0:
                sys.stderr.write('\r%d lines processed...' % i)
            s = l.split()
            embeddings[str(s[0], 'utf-8')] = np.array([float(s[i]) for i in range(1, len(s))])
    with open('data/MEG/glove.obj', 'wb') as f:
        pickle.dump(embeddings, f)

sys.stderr.write('\nEmbeddings loaded.\n')




# OCCUPATIONS

words = []
wordset = set()
freqs = []
feats = []
with open('data/MEG/occupations_1w.txt', 'r') as f:
    for l in f:
        w = l.strip()
        if w in embeddings and w in frequencies:
            words.append(w)
            freqs.append(frequencies[w])
            feats.append(embeddings[w])
            wordset.add(w)

with open('data/MEG/evs_occupations.txt', 'r') as f:
    for l in f:
        w = l.strip()
        if not w in wordset and w in embeddings and w in frequencies:
            words.append(w)
            freqs.append(frequencies[w])
            feats.append(embeddings[w])


sys.stderr.write('Clustering %d occupations...\n' % len(words))

words = np.array(words)
freqs = np.array(freqs)
X = np.stack(feats, axis=0)

m = ALGO(n_clusters=K, n_init=100, random_state=1, n_jobs=1)
m.fit(X)
c = m.predict(X)
_X = m.transform(X)

clust = sorted(list(np.unique(c)))
if not os.path.exists('output/MEG/stim'):
    os.makedirs('output/MEG/stim')
with open('output/MEG/stim/is_a.txt', 'w') as f:
    f.write('Num clusters: %d\n' % len(clust))
    for l in clust:
        ix = c == l
        words_cur = words[ix]
        dist = _X[ix, l]
        freq = freqs[ix]

        f.write('Cluster %s:\n' % (l + 1))
        for w, r, d in sorted(zip(list(words_cur), freq, dist), key=lambda x: x[2]):
            f.write('%s %s %s\n' % (w, r, d))
        f.write('\n')



# NOUNS

nouns = pd.read_csv('data/MEG/has_a.csv')
# nouns = nouns[nouns.prefix.isin(['have a', 'has a', 'had a', 'own a', 'owns a'])]
nouns = nouns[nouns.prefix.isin(['own a', 'owns a'])]
thresh = np.quantile(nouns.logprob, 0.5)
nouns = nouns[nouns.logprob > thresh]
nouns = set(nouns.word)

words = []
freqs = []
feats = []
for w in nouns:
    if w in embeddings and w in frequencies and w in cpdist and cpdist[w].max() == 'NN':
        words.append(w)
        freqs.append(frequencies[w])
        feats.append(embeddings[w])

sys.stderr.write('Clustering %d common nouns...\n' % len(words))

words = np.array(words)
freqs = np.array(freqs)
X = np.stack(feats, axis=0)

m = ALGO(n_clusters=K, n_init=100, random_state=2, n_jobs=1)
m.fit(X)
c = m.predict(X)
_X = m.transform(X)

clust = sorted(list(np.unique(c)))
with open('output/MEG/stim/has_a.txt', 'w') as f:
    f.write('Num clusters: %d\n' % len(clust))
    for l in clust:
        ix = c == l
        words_cur = words[ix]
        freq = freqs[ix]
        dist = _X[ix, l]

        f.write('Cluster %s:\n' % (l + 1))
        for w, r, d in sorted(zip(list(words_cur), freq, dist), key=lambda x: x[2]):
            f.write('%s %s %s\n' % (w, r, d))
        f.write('\n')




# VERBS

verbs = pd.read_csv('data/MEG/likes_to.csv')
verbs = verbs[verbs.prefix.isin(['like to', 'likes to', 'liked to'])]
thresh = np.quantile(verbs.logprob, 0.5)
verbs = verbs[verbs.logprob > thresh]
verbs = set(verbs.word)

words = []
freqs = []
feats = []
for w in verbs:
    if w in embeddings and w in frequencies and w in cpdist and cpdist[w].max() in ['VB', 'VBP']:
        words.append(w)
        freqs.append(frequencies[w])
        feats.append(embeddings[w])

sys.stderr.write('Clustering %d verbs...\n' % len(words))

words = np.array(words)
freqs = np.array(freqs)
X = np.stack(feats, axis=0)
m = ALGO(n_clusters=K, n_init=100, random_state=3, n_jobs=1)
m.fit(X)
c = m.predict(X)
_X = m.transform(X)

clust = sorted(list(np.unique(c)))
with open('output/MEG/stim/likes_to.txt', 'w') as f:
    f.write('Num clusters: %d\n' % len(clust))
    for l in clust:
        ix = c == l
        words_cur = words[ix]
        freq = freqs[ix]
        dist = _X[ix, l]

        f.write('Cluster %s:\n' % (l + 1))
        for w, r, d in sorted(zip(list(words_cur), freq, dist), key=lambda x: x[2]):
            f.write('%s %s %s\n' % (w, r, d))
        f.write('\n')

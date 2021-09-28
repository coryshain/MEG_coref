import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

K = 150

algo = MiniBatchKMeans
viz = PCA

df = pd.read_csv('data/MEG/subtlex.tokmeasures', sep=' ')
df = df.dropna()

lab = df.word.values
X = df[[x for x in df.columns if x.startswith('d')]].values

m = algo(n_clusters=K, n_init=100)
m.fit(X)

sel = (df.freqcount >= 100) & (df.freqcount <= 20000)
_X = X[sel]
_lab = lab[sel]

c = m.predict(_X)

# dr = viz(n_components=2)
# _X = dr.fit_transform(X)
#
# plt.scatter(_X[:,0], _X[:,1], c=c, cmap='gist_rainbow')
# plt.savefig('subtlex_clusters.png')

clust = list(np.unique(c))
print('Num clusters: %d' % len(clust))
for l in clust:
    ix = c == l
    words = _lab[ix]

    print('Cluster %s:' % l)
    for w in sorted(list(words)):
        print(w)
    print()






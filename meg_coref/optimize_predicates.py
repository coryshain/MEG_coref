import os
import random
random.seed(11112020)
import numpy as np
np.random.seed(11112020)
import pandas as pd

embeddings = {}
pred_types = {}
freqs = {}
with open('output/stim/all_preds_filtered.txt', 'r') as f:
    for i, l in enumerate(f):
        s = l.split()
        word = s[0]
        pred_type = s[1]
        freq = int(s[2])
        feats = np.array([float(s[i]) for i in range(3, len(s))])
        embeddings[word] = feats
        pred_types[word] = pred_type
        freqs[word] = freqs

words = list(embeddings.keys())
words = sorted(words)
is_a = [x for x in words if pred_types[x] == 'is-a']
owns_a = [x for x in words if pred_types[x] == 'owns-a']
likes_to = [x for x in words if pred_types[x] == 'likes-to']
feats = np.array([embeddings[w] for w in words])
R = np.corrcoef(feats, rowvar=True)
R = pd.DataFrame(R, columns=words, index=words)

samples = 10
runs = 10

for s in range(samples):
    _max = np.inf
    for r in range(runs):
        sel = set(np.random.choice(is_a, size=5, replace=False)) | \
              set(np.random.choice(owns_a, size=5, replace=False)) | \
              set(np.random.choice(likes_to, size=5, replace=False))

        _R = R.loc[sel, sel] * (1 - np.identity(15))
        converged = False
        while not converged:
            converged = True
            a, b = np.unravel_index(np.argmax(_R), (15, 15))
            w1 = _R.index[a]
            w2 = _R.columns[b]
            _words = np.random.permutation(words)
            for w in _words:
                if w not in sel:
                    for x in (w1, w2):
                        if w != x:
                            pred_type = pred_types[x]
                            if pred_types[w] == pred_type:
                                _sel = (sel - {x}) | {w}
                                __R = R.loc[_sel, _sel] * (1 - np.identity(15))
                                c, d = np.unravel_index(np.argmax(__R), (15, 15))
                                if __R.iloc[c, d] < _R.iloc[a, b]:
                                    _R = __R
                                    sel = _sel
                                    a, b = c, d
                                    w1 = _R.index[a]
                                    w2 = _R.columns[b]
                                    converged = False
                                    break

        if _R.values.max() < _max:
            out = _R

    if not os.path.exists('stim_gen'):
        os.makedirs('stim_gen')

    with open('stim_gen/preds_optimized.txt', 'a') as f:
        max_sim = out.values.max()
        min_sim = out.values.min()
        sims = []
        for i in range(15):
            for j in range(i+1, 15):
                sims.append(out.iloc[i,j])
        mean_sim = np.array(sims).mean()

        f.write('-' * 50 + '\n')
        f.write('Sample %s\n' % (s + 1))
        f.write('Mean sim:          %.4f\n' % mean_sim)
        f.write('Min sim:           %.4f\n' % min_sim)
        f.write('Maximum sim:       %.4f\n' % max_sim)
        f.write('Most similar pair: %s, %s\n' % (w1, w2))
        f.write('Predicates:\n')
        for x in out.index:
            f.write('%s\n' % x)
        f.write('\n')

import sys
import os
import string
import pickle
import numpy as np
import pandas as pd
import textgrid
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


def get_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


wiki_map = {
    'wiki_1': 'actor',
    'wiki_2': 'bagpipe',
    'wiki_3': 'bellhop',
    'wiki_4': 'crochet',
    'wiki_5': 'flirt',
    'wiki_6': 'limousine',
    'wiki_7': 'ophthalmologist',
    'wiki_8': 'paint',
    'wiki_9': 'paralegal',
    'wiki_10': 'pub',
    'wiki_11': 'read',
    'wiki_12': 'retiree',
    'wiki_13': 'sail',
    'wiki_14': 'salamander',
    'wiki_15': 'treadmill'
}

translate_table = dict((ord(char), None) for char in '.,!?;:()[]{}"\'')

lemmatizer = WordNetLemmatizer()

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

sys.stderr.write('Embeddings loaded.\n')

textgrid_dir = 'alignment'

textgrid_paths = [os.path.join(textgrid_dir, x) for x in os.listdir(textgrid_dir) if x.endswith('TextGrid')]

word_times = []
for textgrid_path in textgrid_paths:
    tg = textgrid.TextGrid.fromFile(textgrid_path)
    condition = os.path.basename(textgrid_path)[:-9]
    condition = wiki_map.get(condition, condition)
    for x in tg[0]: # Word tier
        word = x.mark
        if word:
            onset = x.minTime
            offset = x.maxTime
            word_times.append((word, condition, onset, offset))
word_times = pd.DataFrame(word_times, columns=['word', 'condition', 'word_onset_time', 'word_offset_time'])
word_times['word_pos'] = word_times.groupby('condition').cumcount() + 1
word_times['time'] = word_times['word_onset_time']

csv_paths = [
    'expt/stim/4sent.csv',
    'expt/stim/wiki.csv'
]

word_feats = []
for csv_path in csv_paths:
    if 'wiki' in csv_path:
        prefix = 'wiki_'
    else:
        prefix = '4sent_'
    items = pd.read_csv(csv_path)
    condition = prefix + (items.ItemID).astype(str)
    condition = condition.map(lambda x: wiki_map.get(x, x))
    items['condition'] = condition

    _items = items.Item.values
    _lemmas = []
    for s in _items:
        s = s.translate(translate_table).replace('mis-filed', 'misfiled')
        s = s.split()
        _lemmas += [lemmatizer.lemmatize(x, get_pos(t)) for x, t in nltk.pos_tag(s)]

    items.Item = items.Item.str.split()
    items = items.explode('Item').reset_index(drop=True)
    items['word'] = items['Item']
    items['lemma'] = _lemmas
    _emb = np.stack([
        embeddings[x] for x in items.lemma.values
    ], axis=0)
    for i in range(_emb.shape[1]):
        items['d%03d' % (i + 1)] = _emb[:, i]

    del items['Item']
    word_feats.append(items)

word_feats = pd.concat(word_feats, axis=0).reset_index(drop=True)
word_feats['word_pos'] = word_feats.groupby('condition').cumcount() + 1
word_feats = pd.merge(word_feats, word_times[['condition', 'word_pos', 'time', 'word_onset_time', 'word_offset_time']], on=['condition', 'word_pos'])
word_feats.condition = word_feats.condition.str.replace('_', '')

extra_word_feats = pd.read_csv('annotations/word_features.csv')
extra_word_feats['word_cat'] = extra_word_feats['word_cat'].fillna('other')
extra_word_feats['attr_cat'] = extra_word_feats['attr_cat'].fillna('other')
del extra_word_feats['ItemID']
del extra_word_feats['word']
word_feats = pd.merge(word_feats, extra_word_feats, on=['condition', 'word_pos'])
word_feats['lemma'] = word_feats.apply(lambda x: x.attr_cat if x.word_cat == 'attr' else x.lemma, axis=1)

cols = [
    'ItemID', 'Voice', 'condition', 'word', 'lemma', 'word_pos', 'time', 'word_onset_time', 'word_offset_time', 'word_cat', 'attr_cat', 'split_val'
] + ['d%03d' % (i + 1) for i in range(300)]

word_feats = word_feats[cols]

word_feats.to_csv('meg_coref_stim_by_word.csv', index=False)
import numpy as np
import pandas as pd

# 150 ethnically-balanced female and male names
names = pd.read_csv('stim_gen/names_newman.csv')

female = names[names.Gender == 'FEMALE'][['Name', 'Gender']].reset_index(drop=True)
male = names[names.Gender == 'MALE'][['Name', 'Gender']].reset_index(drop=True)

# 15 predicates from 3 equal groups
owns_a = [
    'owns a pub',
    'owns a salamander',
    'owns a treadmill',
    'owns a bagpipe',
    'owns a limousine',
]

is_a = [
    'is a retiree',
    'is an actor',
    'is a paralegal',
    'is an ophthalmologist',
    'is a bellhop',
]

likes_to = [
    'likes to crochet',
    'likes to flirt',
    'likes to sail',
    'likes to paint',
    'likes to read',
]

predicates = owns_a + is_a + likes_to
predicate_types = ['owns_a'] * 5 + ['is_a'] * 5 + ['likes_to'] * 5

predicates = pd.DataFrame({
    'Pred': predicates,
    'PredType': predicate_types
})

# 15 implicit causality verbs in 3 bins
ic_src = pd.read_csv('stim_src/ic.csv')

# Randomly assign predicates to names such that no pairs contain the same predicate
# and all predicates are equally represented
predicates_female = []
predicates_male = []
for i in range(10):
    _predicates_female = predicates.sample(frac=1).reset_index(drop=True)
    _predicates_male = predicates.sample(frac=1).reset_index(drop=True)
    while np.any(_predicates_female.Pred.values == _predicates_male.Pred.values):
        _predicates_male = predicates.sample(frac=1)
    predicates_female.append(_predicates_female)
    predicates_male.append(_predicates_male)

predicates_female = pd.concat(predicates_female).reset_index(drop=True)
predicates_male = pd.concat(predicates_male).reset_index(drop=True)

female = pd.concat([female, predicates_female], axis=1)
male = pd.concat([male, predicates_male], axis=1)

# Randomly decide the name order (female or male first)
order = np.concatenate([np.zeros(75), np.ones(75)]).astype(bool)
np.random.shuffle(order)

ix = np.concatenate([
    female.index[order].to_numpy(),
    female.index[~order].to_numpy(),
], axis=0)

f_first = pd.concat([
    female[order].rename(lambda x: x + '1', axis=1),
    male[order].rename(lambda x: x + '2', axis=1),
], axis=1)

m_first = pd.concat([
    male[~order].rename(lambda x: x + '1', axis=1),
    female[~order].rename(lambda x: x + '2', axis=1),
], axis=1)

stubs = pd.concat([f_first, m_first], axis=0).sort_index()

# Randomly decide whether to reverse the order of names in S1, S2
reversed = np.concatenate([np.zeros(75), np.ones(75)]).astype(bool)
np.random.shuffle(reversed)
stubs['S1SubjReversed'] = reversed
stubs['S1'] = stubs.apply(lambda x: '%s %s and %s %s' % (x.Name2, x.Pred2, x.Name1, x.Pred1) if x.S1SubjReversed else '%s %s and %s %s' % (x.Name1, x.Pred1, x.Name2, x.Pred2), axis=1)
np.random.shuffle(reversed)
stubs['S2SubjReversed'] = reversed
stubs['S2Subj'] = stubs.apply(lambda x: '%s and %s' % (x.Name2, x.Name1) if x.S2SubjReversed else '%s and %s' % (x.Name1, x.Name2), axis=1)

# Randomly assign implicit causality verbs to rows, avoiding any repeat orders
ic = []
for i in range(10):
    ident = True
    while ident:
        _ic = ic_src.sample(frac=1).reset_index(drop=True)
        ident = False
        for x in ic:
            if np.all(x.Verb.values == _ic.Verb.values):
                ident = True
                break
        ic.append(_ic)
ic = pd.concat(ic, axis=0).reset_index(drop=True)
stubs = pd.concat([stubs, ic], axis=1).reset_index(drop=True)

# Shuffle names/genders to randomize continuation-type assignment
stubs = stubs.sample(frac=1)

# Assign continuation types and pronouns
stubs['ICIX'] = stubs.groupby(['Verb']).cumcount()
stubs['ContType'] = stubs.apply(
    lambda x: 'NP1' if ((x.NPBiasCat == 'NP1' and x.ICIX <8) or (x.NPBiasCat == 'NP2' and x.ICIX >= 8) or (x.NPBiasCat == 'neutral' and x.ICIX >= 5)) else 'NP2',
    axis=1
)
stubs['ICPro'] = stubs.apply(
    lambda x: 'she' if ((x.Gender1 == 'FEMALE' and x.ContType == 'NP1') or (x.Gender1 == 'MALE' and x.ContType == 'NP2')) else 'he',
    axis=1
)
del stubs['ICIX']

# Group by verb
stubs = stubs.sort_values(['NPBias', 'Verb'])

# Add todo columns
stubs['Filler'] = 'FILLER'
stubs['AdvP'] = 'ADVP'
stubs['Tense'] = 'TENSE'
stubs['Continuation'] = 'CONT'

# Piece together sentence stubs
stubs['Item'] = stubs.apply(
    lambda x: '%s. %s %s. %s %s.CONJ %s %s because %s %s' % (x.S1, x.S2Subj, x.Filler, x.Name1, x.Verb, x.Name2, x.AdvP, x.ICPro, x.Continuation),
    axis=1
)

# Save output
stubs.to_csv('stim_gen/items.csv', index=False)
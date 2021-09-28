import sys
import re
import pickle
import pandas as pd

is_a_re = re.compile('([Aa]m|[Ii]s|[Aa]re|[Ww]as|[Ww]ere|[Bb]e|) a ')
has_a_re = re.compile('([Hh]ave|[Hh]as|[Hh]ad|[Oo]wn|[Oo]wns|[Oo]wned) a ')
likes_to_re = re.compile('([Ll]ike|[Ll]ikes|[Ll]iked) to ')

has_a = []
is_a = []
likes_to = []

i = 0
   
in_3_grams = False
with open('/data/compling/kenlm_models/gigaword.3.kenlm', 'r') as f:
    for l in f:
        if not in_3_grams:
            if l.startswith('\\3-grams'):
                in_3_grams = True
        elif l[0] == '-': # Minus sign before all logprobs
            p, g = l.strip().split('\t')
            if i % 1000000 == 0:
                sys.stderr.write('\rNum processed = %d' % i)
                sys.stderr.flush()
            if has_a_re.match(g):
            #if g.startswith('has a '):
                s = g.split()
                w = s[-1]
                c = ' '.join(s[:-1])
                has_a.append((w, c, p))
            elif is_a_re.match(g):
            #elif g.startswith('is a '):
                s = g.split()
                w = s[-1]
                c = ' '.join(s[:-1])
                is_a.append((w, c, p))
            elif likes_to_re.match(g):
            #elif g.startswith('likes to '):
                s = g.split()
                w = s[-1]
                c = ' '.join(s[:-1])
                likes_to.append((w, c, p))
            i += 1

sys.stderr.write('\n')
sys.stderr.flush()

has_a = sorted(has_a, key=lambda x: x[-1])
is_a = sorted(is_a, key=lambda x: x[-1])
likes_to = sorted(likes_to, key=lambda x: x[-1])

has_a = pd.DataFrame(has_a, columns=['word', 'prefix', 'logprob'])
is_a = pd.DataFrame(is_a, columns=['word', 'prefix', 'logprob'])
likes_to = pd.DataFrame(likes_to, columns=['word', 'prefix', 'logprob'])

has_a.to_csv('has_a.csv', index=False)
is_a.to_csv('is_a.csv', index=False)
likes_to.to_csv('likes_to.csv', index=False)



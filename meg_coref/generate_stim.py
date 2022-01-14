import random
import yaml
import pandas as pd
import argparse

'''
N1
N2
PRO1
PROACC1
PROPOSS1
PROREFL1
PRO2
PROACC2
PROPOSS2
PROREFL2
'''

pronoun_map_male = {
    'PRO%d': 'he',
    'PROACC%d': 'him',
    'PROPOSS%d': 'his',
    'PROREFL%d': 'himself'
}

pronoun_map_female = {
    'PRO%d': 'she',
    'PROACC%d': 'her',
    'PROPOSS%d': 'her',
    'PROREFL%d': 'herself'
}


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
        Generate stimuli for MEG experiment.
    ''')
    argparser.add_argument('-n', '--nsample', default=1, type=int, help='Number of items to generate.')
    args = argparser.parse_args()

    with open('output/MEG/stim/stim.txt', 'r') as f:
        stim_src = yaml.load(f)

    gender = ['M', 'F']
    pronoun = {
        'M': {
            'PRO': 'he',
            'PROACC': 'him',
            'PROPOSS': 'his'
        },
        'F': {
            'PRO': 'she',
            'PROACC': 'her',
            'PROPOSS': 'her'
        }
    }
    ethnicities = ['asian', 'black', 'hispanic', 'white']
    names = stim_src['name']
    predicates = stim_src['predicate']
    predicate_types = ['owns_a', 'is_a', 'likes_to']
    fillers = stim_src['filler']

    df = pd.read_csv('../experiments/MEG_coref/stimuli/meg_coref_stim_src.csv')
    main_clauses = df.main_clause.values.tolist()
    advps = df.advp.values.tolist()
    complements = df.complement.values.tolist()

    n_stim = len(df)

    name1 = []
    name1_gender = []
    name1_ethnicity = []
    name2 = []
    name2_gender = []
    name2_ethnicity = []
    pred1 = []
    pred1_type = []
    pred2 = []
    pred2_type = []
    filler = []
    filler_order = []
    complement = []
    critical = []
    full = []
    for i in range(n_stim):
        em = ethnicities[random.randint(0, 3)]
        n = len(names['F'][em])
        female_name = names['F'][em][random.randint(0, n - 1)]

        ef = ethnicities[random.randint(0, 3)]
        n = len(names['M'][ef])
        male_name = names['M'][ef][random.randint(0, n - 1)]

        s = random.randint(0, 1)
        if s:
            _name1 = female_name
            _name2 = male_name
            _ethnicity1 = ef
            _ethnicity2 = em
            gender1 = 'female'
            gender2 = 'male'
        else:
            _name1 = male_name
            _name2 = female_name
            _ethnicity1 = em
            _ethnicity2 = ef
            gender1 = 'male'
            gender2 = 'female'
        name1.append(_name1)
        name1_gender.append(gender1)
        name1_ethnicity.append(_ethnicity1)
        name2.append(_name2)
        name2_gender.append(gender2)
        name2_ethnicity.append(_ethnicity2)

        p = predicate_types[random.randint(0, 2)]
        pred1_type.append(p)
        n = len(predicates[p])
        _pred1 = predicates[p][random.randint(0, n - 1)]
        pred1.append(_pred1)

        p = predicate_types[random.randint(0, 2)]
        pred2_type.append(p)
        n = len(predicates[p])
        _pred2 = predicates[p][random.randint(0, n - 1)]
        pred2.append(_pred2)

        s1 = '%s %s and %s %s.' % (_name1, _pred1, _name2, _pred2)

        o = random.randint(0, 1)
        filler_order.append(o)
        filler_pred = fillers[random.randint(0, len(fillers) - 1)]
        if o:
            _filler = '%s and %s %s.' % (_name1, _name2, filler_pred)
        else:
            _filler = '%s and %s %s.' % (_name2, _name1, filler_pred)
        filler.append(_filler)

        _main_clause = main_clauses[i].replace('N1', _name1).replace('N2', _name2)
        _advp = advps[i]
        _complement = complements[i].replace('N1', _name1).replace('N2', _name2)
        if gender1 == 'female':
            pmap1 = pronoun_map_female
            pmap2 = pronoun_map_male
        else:
            pmap1 = pronoun_map_male
            pmap2 = pronoun_map_female

        for x in pmap1:
            _complement = _complement.replace(x % 1, pmap1[x])
        for x in pmap2:
            _complement = _complement.replace(x % 2, pmap2[x])
        complement.append(_complement)

        _critical = '%s %s %s' % (_main_clause, _advp, _complement)
        critical.append(_critical)

        _full = ' '.join([s1, _filler, _critical])
        full.append(_full)

    out = pd.DataFrame({
        'item': full,
        'name1': name1,
        'name1_gender': name1_gender,
        'name1_ethnicity': name1_ethnicity,
        'name2': name2,
        'name2_gender': name2_gender,
        'name2_ethnicity': name2_ethnicity,
        'pred1': pred1,
        'pred1_type': pred1_type,
        'pred2': pred2,
        'pred2_type': pred2_type,
        'filler': filler,
        'filler_order': filler_order,
        'complement_filled': complement,
        'critical': critical
    })

    out = pd.concat([out, df], axis=1)
    out.to_csv('../experiments/MEG_coref/stimuli/meg_coref_stim_sampled.csv', index=False)





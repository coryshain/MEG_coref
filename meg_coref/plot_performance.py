import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from matplotlib import pyplot as plt
import argparse

plt.rcParams["font.family"] = "Arial"


def acc(x):
    return accuracy_score(x.CDRobs, x.CDRpreds)


def f1(x):
    return f1_score(x.CDRobs, x.CDRpreds, average='macro')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
        Plot performance over time.
    ''')
    argparser.add_argument('path', help='Path to score table.')
    argparser.add_argument('-M', '--metric', default='f1', help='Metric to use. One of ``acc``, ``f1``.')
    argparser.add_argument('-b', '--baseline', default='uniform', help='Baseline to use. If ``"infer"``, infer from data.')
    argparser.add_argument('-g', '--grouping_variable', default=['subject'], nargs='+', help='Column(s) defining grouping variables within which to compute performance.')
    args = argparser.parse_args()

    path = args.path
    dirpath = os.path.dirname(path)
    metric = args.metric.lower()
    grouping_variable = args.grouping_variable
    baseline = args.baseline

    df = pd.read_csv(path, sep=' ')
    if metric == 'f1':
        fn = f1
    elif metric == 'acc':
        fn = acc
    else:
        raise ValueError('Unrecognized metric: %s' % metric)
    if baseline.lower() == 'infer':
        train_df = pd.read_csv(path.replace('dev', 'train').replace('test', 'train'), sep=' ')
        classes, counts = np.unique(train_df.CDRobs, return_counts=True)
        majority = classes[np.argmax(counts)]
        baseline = fn(pd.DataFrame({'CDRpreds': [majority] * len(df), 'CDRobs': df.CDRobs}))
    elif baseline.lower() == 'uniform':
        baseline = 1 / len(df.CDRobs.unique())
    else:
        baseline = float(baseline)
    perf = df.groupby(['tdelta'] + grouping_variable)[['CDRpreds', 'CDRobs']].apply(f1).reset_index()

    mean = []
    plt.axvline(0., linewidth=2, color='k')
    plt.axhline(baseline, linewidth=2, color='r')
    for k, _df in perf.groupby(grouping_variable):
        x = _df.tdelta
        y = _df[0]
        mean.append(y)
        plt.plot(x, y, color=(0.6, 0.6, 0.6), linewidth=1)
    mean = np.stack(mean, axis=0).mean(axis=0)
    plt.plot(x, mean, color='blue', linewidth=2)
    if metric == 'f1':
        plt.ylabel('F1')
    elif metric == 'acc':
        plt.ylabel('% Correct')
    else:
        raise ValueError('Unrecognized metric: %s' % metric)
    plt.xlabel('Time')
    plt.savefig(path[:-4] + '_%s.png' % metric)

import os
import math
import shutil
import pickle
import re
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA, PCA
from sklearn.pipeline import Pipeline
import argparse
from matplotlib import pyplot as plt

from .util import *
from .classifiers import *
from .nn import DNN

channel_matcher = re.compile('(MEG\d\d\d\d)')

MAX_N_RESAMP = 100


glove_matcher = re.compile('d\d{3}')
iter_matcher = re.compile('i(\d+)')
fold_matcher = re.compile('f(\d+)')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Run cross-validated decoding.
    ''')
    argparser.add_argument('config', help='Path to config file containing decoding settings.')
    args = argparser.parse_args()

    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=Loader)

    confint = config.get('confint', 95)
    onset = config.get('onset', 0.5)
    downsample_by = config.get('downsample_by', 10)
    outdir = os.path.normpath(config.get('outdir', './results'))

    results = {
        'acc': [],
        'f1': []
    }
    results_f1 = []
    chance = {
        'acc': None,
        'f1': None,
    }
    for i in [iter_matcher.match(x).group(1) for x in os.listdir(outdir) if iter_matcher.match(x)]:
        iterdir = os.path.join(outdir, 'i%s' % i)
        for j in [fold_matcher.match(x).group(1) for x in os.listdir(iterdir) if fold_matcher.match(x)]:
            folddir = os.path.join(iterdir, 'f%s' % j)
            respath = os.path.join(folddir, 'results.obj')
            if os.path.exists(respath):
                with open(respath, 'rb') as f:
                    _results = pickle.load(f)
                results['acc'].append(_results['acc'])
                results['f1'].append(_results['f1'])
                if chance['acc'] is None:
                    chance['acc'] = _results['chance_acc']
                if chance['f1'] is None:
                    chance['f1'] = _results['chance_f1']

    results['acc'] = np.stack(results['acc'], axis=0)
    results['f1'] = np.stack(results['f1'], axis=0)

    for score in results:
        _results = results[score] * 100  # Place scores on 0-100 for readability
        _chance = chance[score] * 100  # Place scores on 0-100 for readability
        ntime = _results.shape[1]

        mean = _results.mean(axis=0)

        a = 100 - confint

        lb, ub = np.percentile(_results, (a / 2, 100 - (a / 2)), axis=0)
        x = np.linspace(0 - onset, ntime / 1000 * downsample_by - onset, ntime)

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.xlim((x.min(), x.max()))
        plt.axvline(0, color='k')
        if score in ('acc', 'f1'):
            plt.axhline(_chance, color='r')
        else:
            raise ValueError('Unrecognized scoring function: %s' % score)
        plt.fill_between(x, lb, ub, alpha=0.2)
        plt.plot(x, mean)
        plt.xlabel('Time (s)')
        if score == 'acc':
            plt.ylabel('% Correct')
        elif score == 'f1':
            plt.ylabel('F1')
        else:
            raise ValueError('Unrecognized scoring function: %s' % score)
        plt.tight_layout()

        if not os.path.exists(outdir):
            os.makedirs(outdir)
        plt.savefig(os.path.join(outdir, 'perf_plot_%s.png' % score))

        plt.close('all')
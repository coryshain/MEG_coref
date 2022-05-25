import os
import shutil
import pickle
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import pandas as pd
import argparse

from .util import *


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Run cross-validated decoding.
    ''')
    argparser.add_argument('config', help='Path to config file containing decoding settings.')
    argparser.add_argument('-f', '--force_resample', action='store_true', help='Force resampling of CV partition indices, even if samples already exist. Otherwise, partition indices will not be overwritten.')
    args = argparser.parse_args()

    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=Loader)
    force_resample = args.force_resample

    paths = config['paths']
    label_field = config.get('label_field', 'attr_cat')
    filters = config.get('filters', {})
    powerband = config.get('powerband', None)
    downsample_by = config.get('downsample_by', 1)
    nfolds = config.get('nfolds', 5)
    assert nfolds > 1, "nfolds must be >= 2."
    niter = config.get('niter', 10)
    separate_subjects = config.get('separate_subjects', True)
    combine_subjects = config.get('combine_subjects', True)
    outdir = config.get('outdir', './results')
    outdir = os.path.normpath(outdir)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if not os.path.normpath(os.path.realpath(config_path)) == os.path.normpath(os.path.realpath(outdir + '/config.ini')):
        shutil.copy2(config_path, outdir + '/config.ini')

    data = []
    for dirpath in paths:
        dirpath = os.path.normpath(dirpath)
        stderr('Loading %s...\n' % dirpath)
        filename = 'data_d%d_p%s.obj' % (downsample_by, '%s-%s' % tuple(powerband) if powerband else 'None')
        cache_path = os.path.join(dirpath, filename)
        with open(cache_path, 'rb') as f:
            data_src = pickle.load(f)
        _data = data_src['data']
        _labels = data_src['labels']

        _label_df = pd.DataFrame(_labels)
        if 'count' not in _label_df.columns:
            labs, ix, counts = np.unique(_label_df[label_field].values, return_inverse=True, return_counts=True)
            counts = counts[ix]
            _label_df['labcount'] = counts
        _filter_mask = compute_filter_mask(_label_df, filters)

        _data = _data[_filter_mask]

        data.append(_data)

    data = np.concatenate(data, axis=0)

    compile_cv_ix(data, outdir, niter=niter, nfolds=nfolds)

import os
import shutil
import pickle
import re
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.decomposition import FastICA, PCA
import argparse
from matplotlib import pyplot as plt

from .util import *
from .classifiers import *
from .nn import DNN

channel_matcher = re.compile('(MEG\d\d\d\d)')

MAX_N_RESAMP = 100


glove_matcher = re.compile('d\d{3}')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Run cross-validated decoding.
    ''')
    argparser.add_argument('config', help='Path to config file containing decoding settings.')
    argparser.add_argument('-f', '--force_reprocess', action='store_true', help='Force data reprocessing, even if a data cache exists.')
    argparser.add_argument('-c', '--cpu_only', action='store_true', help='Force CPU implementation if GPU available.')
    args = argparser.parse_args()

    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=Loader)
    force_reprocess = args.force_reprocess
    if args.cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    paths = config['paths']
    label_field = config.get('label_field', 'attr_cat')
    filters = config.get('filters', {})
    powerband = config.get('powerband', None)
    downsample_by = config.get('downsample_by', 1)
    bysensor = config.get('bysensor', False)
    nfolds = config.get('nfolds', 5)
    assert nfolds > 1, "nfolds must be >= 2."
    resamp = config.get('resamp', True)
    niter = config.get('niter', 10)
    nlabperfold = config.get('nlabperfold', 100)
    clstype = config.get('clstype', 'MaxCorrelation')
    bytimestep = clstype.lower() != 'dnn'
    predict_single_trial = config.get('predict_single_trial', False)
    use_glove = config.get('use_glove', False)
    glove_cls = ('Ridge',)
    assert not use_glove or clstype in glove_cls, 'GloVe-based decoding is only compatible with the following classifier types: %s' % ', '.join(glove_cls)
    nlag = config.get('nlag', 0)
    assert not bysensor or nlag, 'If bysensor is used, nlag must be positive (otherwise, would be correlating scalars)'
    separate_subjects = config.get('separate_subjects', True)
    combine_subjects = config.get('combine_subjects', True)
    zscore_sensors = config.get('zscore_sensors', False)
    tanh_transform = config.get('tanh_transform', False)
    rank_transform = config.get('rank_transform', False)
    k_feats = config.get('k_feats', 0)
    k_pca = config.get('k_pca', 0)
    k_pca_glove = config.get('k_pca_glove', 0)
    k_pca_drop = config.get('k_pca_drop', 0)
    k_ica = config.get('k_ica', 0)
    denoise = config.get('denoise', False)
    dropout = config.get('dropout', 0)
    score = config.get('score', 'acc')
    confint = config.get('confint', 95)
    onset = config.get('onset', 0.5)
    outdir = config.get('outdir', './results')

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if not os.path.normpath(os.path.realpath(config_path)) == os.path.normpath(os.path.realpath(outdir + '/config.ini')):
        shutil.copy2(config_path, outdir + '/config.ini')

    if bysensor:
        sensor_indices = list(zip(range(306), range(1, 307)))
    else:
        sensor_indices = [(0, 306)]

    data = []
    labels = []
    filter_masks = []
    names = []
    ntime = None
    for dirpath in paths:
        dirpath = os.path.normpath(dirpath)
        name = os.path.basename(os.path.dirname(dirpath))
        stderr('Loading %s...\n' % dirpath)
        filename = 'data_d%d_p%s.obj' % (downsample_by, '%s-%s' % tuple(powerband) if powerband else 'None')
        cache_path = os.path.join(dirpath, filename)
        with open(cache_path, 'rb') as f:
            data_src = pickle.load(f)
        _data = data_src['data']
        _labels = data_src['labels']
        meta = data_src['meta']

        _filter_mask = compute_filter_mask(pd.DataFrame(_labels), filters)
        cols = [label_field]
        if use_glove:
            cols += sorted([x for x in _labels if glove_matcher.match(x)])
        _labels = np.stack([_labels[x] for x in cols], axis=1)

        data.append(_data)
        labels.append(_labels)
        filter_masks.append(_filter_mask)
        names.append(name)

    _data = []
    _labels = []
    _filter_masks = []
    _names = []
    if separate_subjects:
        _data += data
        _labels += labels
        _filter_masks += filter_masks
        _names += names
    if combine_subjects:
        _data.append(np.concatenate(data, axis=0))
        _labels.append(np.concatenate(labels, axis=0))
        _filter_masks.append(np.concatenate(filter_masks, axis=0))
        _names.append('combined')
    data = _data
    labels = _labels
    filter_masks = _filter_masks
    names = _names

    for __data, _labels, _filter_mask, _name in zip(data, labels, filter_masks, names):
        for sensor_ix in sensor_indices:
            if len(sensor_indices) == 1:
                sensor_name = 'allsensors'
            else:
                sensor_name = '%03d' % (sensor_ix[0] + 1)
            stderr('Regressing %s...\n' % _name)
            _data = __data[:, sensor_ix[0]:sensor_ix[1]]
            labs, counts = np.unique(_labels[:, 0], return_counts=True)
            _nclass = len(labs)
            majority_class_prop = counts.max() / len(_labels)
            # _data = convolve(_data, [[[0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25]]], mode='same', method='direct')
            if zscore_sensors:
                stderr('Z-scoring over time...\n')
                _data = zscore(_data, axis=2)
            if tanh_transform:
                stderr('Tanh-transforming...\n')
                _data = np.tanh(_data)
            if rank_transform:
                stderr('Rank-transforming over sensors...\n')
                _data = rankdata(_data, axis=1)
            if k_pca or k_ica:
                __data = _data.transpose([0, 2, 1])
                b, t = __data.shape[:2]
                __data = __data.reshape((-1, _data.shape[1]))
                if k_pca:
                    stderr('Selecting top %d principal components...\n' % k_pca)
                    __data = zscore(__data, axis=0)
                    pca = PCA(k_pca)
                    _data = pca.fit_transform(__data)
                    _data = _data.reshape((b, t, k_pca))
                    _data = _data.transpose([0, 2, 1])
                    _data = _data[:, k_pca_drop:, :]
                    stderr('  %.2f%% variance explained\n' % (pca.explained_variance_ratio_[k_pca_drop:].sum() * 100))
                if k_ica:
                    stderr('Selecting %d independent components...\n' % k_ica)
                    _data = FastICA(k_ica).fit_transform(__data)
                    _data = _data.reshape((b, t, k_ica))
                    _data = _data.transpose([0, 2, 1])
            if use_glove and k_pca_glove:
                stderr('Selecting top %d principal GloVe components...\n' % k_pca_glove)
                __labels = PCA(k_pca_glove).fit_transform(zscore(_labels[:, 1:].astype('float'), axis=0))
                _labels = np.concatenate([_labels[:, :1], __labels], axis=1)
            if nlag:
                stderr('Adding lags...\n')
                n = _data.shape[2]
                _data = np.concatenate([_data[..., i:n - nlag + i] for i in range(nlag + 1)], axis=1)
            if ntime is None:
                ntime = _data.shape[-1]
            results = np.zeros((niter, nfolds, ntime))

            for i in range(niter):
                stderr('Iteration %d/%d' % (i + 1, niter))
                cv_src, labels_src, train_supp_src = partition_cv(_data, _labels, nfolds, filter_mask=_filter_mask)

                for j in range(nfolds):
                    stderr('\n  CV fold %s/%s\n' % (j + 1, nfolds))
                    X_train_src = {
                        lab: np.concatenate([cv_src[k][lab] for k in range(nfolds) if k != j]) for lab in cv_src[j]
                    }
                    y_train_src = {lab: labels_src[lab] for lab in X_train_src}
                    for k in train_supp_src:
                        if k in X_train_src:
                            X_train_src[k] = np.concatenate([X_train_src[k], train_supp_src[k]])
                        else:
                            X_train_src[k] = train_supp_src[k]
                        if k not in y_train_src:
                            y_train_src[k] = labels_src[k]
                    X_train = []
                    y_train = []
                    mu_train = []
                    if nlabperfold:
                        nlab = nlabperfold * (nfolds - 1)
                    else:
                        nlab = None
                    mean_by_lab = {}
                    stderr('    Processing data\n')
                    for lab in X_train_src:
                        _X = X_train_src[lab]
                        mean_by_lab[lab] = _X.mean(axis=0, keepdims=True)
                        if resamp:
                            if nlab:
                                _X_out = np.zeros((nlab, _X.shape[1], _X.shape[2]))
                                for k in range(nlab):
                                    n = min(len(_X), MAX_N_RESAMP)
                                    ix = np.random.choice(np.arange(len(_X)), size=n, replace=False)
                                    __X = _X[ix]
                                    a = np.ones(n)
                                    w = np.random.dirichlet(a)  # Sample mixing weights
                                    __X = np.tensordot(w, __X, axes=(0, 0))
                                    _X_out[k] = __X
                                _X = _X_out
                            else:
                                _X = mean_by_lab[lab]
                        else:
                            if nlab:
                                assert len(_X) >= nlab, 'Requesting more data than is contained in the training set. Reduce nfold or nlabperfold, or rerun with --resamp.'
                                _X = np.array_split(_X, nlab, axis=0)
                                _X = np.stack([__X.mean(axis=0) for __X in _X], axis=0)
                        _y = np.repeat(y_train_src[lab][None, ...], len(_X), axis=0)
                        X_train.append(_X)
                        y_train.append(_y)
                        if denoise:
                            _mu = np.repeat(mean_by_lab[lab], len(_X), axis=0)
                            mu_train.append(_mu)

                    X_train = np.concatenate(X_train, axis=0)
                    ix = np.random.permutation(np.arange(len(X_train)))
                    X_train = X_train[ix]
                    y_train = np.concatenate(y_train, axis=0)
                    y_train = y_train[ix]
                    if denoise:
                        mu_train = np.concatenate(mu_train, axis=0)
                        mu_train = mu_train[ix]
                    if use_glove:
                        y_train = y_train[:, 1:].astype(float)  # Drop label dimension, keep glove dimensions

                    if denoise:
                        stderr('    Training denoiser\n')
                        denoiser = DNN(
                            layer_type='cnn',
                            n_layers=1,
                            n_units=100,
                            kernel_width=1,
                            # cnn_activation=None,
                            n_outputs=306,
                            dropout=None,
                            reg_scale=100.,
                            continuous_outputs=True,
                            project=True
                        )
                        X_train = np.transpose(X_train, [0, 2, 1])
                        mu_train = np.transpose(mu_train, [0, 2, 1])
                        denoiser.fit(
                            X_train,
                            mu_train,
                            batch_size=32,
                            epochs=100,
                            validation_split=0.1,
                            callbacks=[tf.keras.callbacks.EarlyStopping(
                                monitor='val_mse', patience=10, restore_best_weights=True
                            )]
                        )
                        X_train = np.transpose(denoiser.predict(X_train, batch_size=32), [0, 2, 1])

                    X_val_src = {lab: cv_src[j][lab] for lab in cv_src[j]}
                    y_val_src = {lab: labels_src[lab] for lab in X_train_src}
                    X_val = []
                    y_val = []
                    nlab = nlabperfold
                    comparison_set = {}
                    for lab in X_val_src:
                        if use_glove:
                            comparison_set[labels_src[lab][0]] = labels_src[lab][1:].astype(float)
                        _X = X_val_src[lab]
                        if not predict_single_trial:
                            if resamp:
                                if nlab:
                                    _X_out = np.zeros((nlab, _X.shape[1], _X.shape[2]))
                                    for k in range(nlab):
                                        n = min(len(_X), MAX_N_RESAMP)
                                        ix = np.random.choice(np.arange(len(_X)), size=n, replace=False)
                                        __X = _X[ix]
                                        a = np.ones(n)
                                        w = np.random.dirichlet(a)  # Sample mixing weights
                                        __X = np.tensordot(w, __X, axes=(0, 0))
                                        _X_out[k] = __X
                                    _X = _X_out
                                else:
                                    _X = _X.mean(axis=0, keepdims=True)
                            else:
                                if nlab:
                                    assert len(_X) >= nlab, 'Requesting more data than is contained in the validation set. Reduce nfold or nlabperfold, or rerun with --resamp.'
                                    _X = np.array_split(_X, nlab, axis=0)
                                    _X = np.stack([__X.mean(axis=0) for __X in _X], axis=0)
                        _y = np.repeat(y_val_src[lab][None, ...], len(_X), axis=0)
                        X_val.append(_X)
                        y_val.append(_y)
                    X_val = np.concatenate(X_val, axis=0)
                    if denoise:
                        X_val = np.transpose(denoiser.predict(np.transpose(X_val, [0, 2, 1]), batch_size=32), [0, 2, 1])
                    y_val = np.concatenate(y_val, axis=0)
                    y_val = y_val[:, 0]  # Drop glove dimensions (or squeeze if no glove), keep label dimension

                    if bytimestep:
                        n_time_train = X_train.shape[-1]
                    else:
                        n_time_train = 1
                    for t in range(n_time_train):
                        stderr('\r    Fitting model %d/%d' % (t + 1, n_time_train))
                        if clstype == 'LogisticRegression':
                            kwargs = {'C': 0.01, 'solver': 'newton-cg'}
                        elif clstype == 'LinearSVM':
                            kwargs = {'C': 0.01}
                        elif clstype == 'SVM':
                            kwargs = {'C': 0.01}
                        elif clstype == 'MaxCorrelation':
                            kwargs = {'k_feats': k_feats, 'dropout': dropout}
                        elif clstype == 'Ridge':
                            kwargs = {'alpha': 100}
                        elif clstype == 'DNN':
                            kwargs = {
                                'layer_type': 'cnn',
                                'learning_rate': 0.0001,
                                'n_units': 256,
                                'n_layers': 1,
                                'kernel_width': 25,
                                'cnn_activation': 'tanh',
                                'reg_scale': None,
                                'dropout': None,
                                'use_glove': use_glove,
                                'n_outputs': _nclass,
                                'layer_normalize': False,
                                'batch_normalize': True,
                            }
                        else:
                            raise ValueError('Unrecognized clstype %s' % clstype)
                        if bytimestep:
                            X_in = X_train[..., t]
                        else:
                            X_in = X_train
                        m = train(X_in, y_train, clstype=clstype, **kwargs)

                        if bytimestep:
                            X_in = X_val[..., t]
                            time_val = [t]
                        else:
                            X_in = X_val
                            time_val = np.arange(X_in.shape[-1])
                        y_pred = classify(m, X_in, argmax=score == 'acc', comparison_set=comparison_set)

                        if bytimestep:
                            y_pred = np.expand_dims(y_pred, 1)
                        for _t in time_val:
                            if bytimestep:
                                _y_pred = y_pred
                            else:
                                _y_pred = y_pred[:, _t]
                            if score == 'acc':
                                s = accuracy_score(y_val, _y_pred)
                            elif score == 'normrank':
                                s = (rankdata(_y_pred, axis=1) - 1) / (_y_pred.shape[1] - 1)
                                labmap = {val: i for i, val in enumerate(m.classes_)}
                                ix = np.array([labmap[_y] for _y in y_val])
                                s = s[np.arange(len(s)), ix].mean()
                            else:
                                raise ValueError('Unrecognized scoring function: %s' % score)
                            results[i, j, _t] = s

                    mean = results[:i+1, :j+1].mean(axis=(0, 1))

                    a = 100 - confint

                    lb, ub = np.percentile(results[:i+1, :j+1], (a / 2, 100 - (a / 2)), axis=(0, 1))
                    lag = nlag / 1000 * downsample_by
                    x = np.linspace(0 - onset + lag, ntime / 1000 * downsample_by - onset + lag, ntime)

                    plt.gca().spines['top'].set_visible(False)
                    plt.gca().spines['right'].set_visible(False)
                    plt.xlim((x.min(), x.max()))
                    plt.axvline(0, color='k')
                    if score == 'acc':
                        plt.axhline(majority_class_prop, color='r')
                    elif score == 'normrank':
                        plt.axhline(0.5, color='r')
                    else:
                        raise ValueError('Unrecognized scoring function: %s' % score)
                    plt.fill_between(x, lb, ub, alpha=0.2)
                    plt.plot(x, mean)
                    plt.xlabel('Time (s)')
                    if score == 'acc':
                        plt.ylabel('% Correct')
                    elif score == 'normrank':
                        plt.ylabel('Normalized Rank')
                    else:
                        raise ValueError('Unrecognized scoring function: %s' % score)
                    plt.tight_layout()

                    if not os.path.exists(outdir):
                        os.makedirs(outdir)
                    with open(os.path.join(outdir, 'results_%s_%s.obj' % (_name, sensor_name)), 'wb') as f:
                        pickle.dump(results[:i+1], f)
                    plt.savefig(os.path.join(outdir, 'perf_plot_%s_%s.png' % (_name, sensor_name)))

                    plt.close('all')

                stderr('\n')

            stderr('\n')


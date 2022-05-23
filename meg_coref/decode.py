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
    eval_filters = config.get('eval_filters', {})
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
    glove_cls = ('Ridge', 'DNN')
    assert not use_glove or clstype in glove_cls, 'GloVe-based decoding is only compatible with the following classifier types: %s' % ', '.join(glove_cls)
    nlag = config.get('nlag', 0)
    assert not bysensor or nlag, 'If bysensor is used, nlag must be positive (otherwise, would be correlating scalars)'
    separate_subjects = config.get('separate_subjects', True)
    combine_subjects = config.get('combine_subjects', True)
    zscore_time = config.get('zscore_time', False)
    zscore_sensors = config.get('zscore_sensors', False)
    normalize_sensors = config.get('normalize_sensors', False)
    tanh_transform = config.get('tanh_transform', False)
    rank_transform = config.get('rank_transform', False)
    k_feats = config.get('k_feats', 0)
    k_pca = config.get('k_pca', 0)
    k_pca_glove = config.get('k_pca_glove', 0)
    k_pca_drop = config.get('k_pca_drop', 0)
    k_ica = config.get('k_ica', 0)
    layer_type = config.get('layer_type', 'cnn')
    learning_rate = config.get('learning_rate', 0.0001)
    n_units = config.get('n_units', 128)
    n_layers = config.get('n_layers', 1)
    kernel_width = config.get('kernel_width', 10)
    cnn_activation = config.get('cnn_activation', 'relu')
    reg_scale = config.get('reg_scale', None)
    sensor_filter_scale = config.get('sensor_filter_scale', None)
    dropout = config.get('dropout', None)
    use_resnet = config.get('use_resnet', False)
    use_locally_connected = config.get('use_locally_connected', False)
    batch_normalize = config.get('batch_normalize', False)
    layer_normalize = config.get('layer_normalize', False)
    l2_layer_normalize = config.get('l2_layer_normalize', False)
    project = config.get('project', True)
    contrastive_loss_weight = config.get('contrastive_loss_weight', None)
    n_dnn_epochs = config.get('n_dnn_epochs', 1000)
    dnn_batch_size = config.get('dnn_batch_size', 32)
    denoise = config.get('denoise', False)
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
    eval_filter_masks = []
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

        _label_df = pd.DataFrame(_labels)
        _filter_mask = compute_filter_mask(_label_df, filters)
        _eval_filter_mask = compute_filter_mask(_label_df, eval_filters)
        cols = [label_field]
        if use_glove:
            cols += sorted([x for x in _labels if glove_matcher.match(x)])
        _labels = np.stack([_labels[x] for x in cols], axis=1)

        if zscore_time:
            stderr('Z-scoring over time...\n')
            _data = zscore(_data, axis=2)
        if zscore_sensors:
            stderr('Z-scoring over sensors...\n')
            _data = zscore(_data, axis=1)
        if tanh_transform:
            stderr('Tanh-transforming...\n')
            _data = np.tanh(_data)
        if normalize_sensors:
            stderr('L2 normalizing over sensors...\n')
            n = np.linalg.norm(_data, axis=1, keepdims=True)
            _data /= n
        if rank_transform:
            stderr('Rank-transforming over sensors...\n')
            _ndim = _data.shape[1]
            _mean = (_ndim + 1) / 2
            _sd = np.arange(_ndim).std()
            _data = (rankdata(_data, axis=1) - 1) / _sd

        _data = _data[_filter_mask]
        _data = np.where(np.isfinite(_data), _data, 0.)
        _labels = _labels[_filter_mask]
        _eval_filter_mask = _eval_filter_mask[_filter_mask]

        data.append(_data)
        labels.append(_labels)
        eval_filter_masks.append(_eval_filter_mask)
        names.append(name)

    _data = []
    _labels = []
    _eval_filter_masks = []
    _names = []
    if separate_subjects:
        _data += data
        _labels += labels
        _eval_filter_masks += eval_filter_masks
        _names += names
    if combine_subjects:
        _data.append(np.concatenate(data, axis=0))
        _labels.append(np.concatenate(labels, axis=0))
        _eval_filter_masks.append(np.concatenate(eval_filter_masks, axis=0))
        _names.append('combined')
    data = _data
    labels = _labels
    eval_filter_masks = _eval_filter_masks
    names = _names

    if use_glove and k_pca_glove:
        stderr('Selecting top %d principal GloVe components...\n' % k_pca_glove)
        vocab, ix = np.unique(labels[-1][:, 0], return_index=True)
        _pca_in = labels[-1][ix, 1:]
        glove_pca = Pipeline([('scaler', StandardScaler()), ('pca', PCA(k_pca_glove))])
        glove_pca.fit(_pca_in)

    for __data, _labels, _eval_filter_mask, _name in zip(data, labels, eval_filter_masks, names):
        for sensor_ix in sensor_indices:
            if len(sensor_indices) == 1:
                sensor_name = 'allsensors'
            else:
                sensor_name = '%03d' % (sensor_ix[0] + 1)
            stderr('Regressing %s...\n' % _name)
            _data = __data[:, sensor_ix[0]:sensor_ix[1]]
            labs, counts = np.unique(_labels[:, 0], return_counts=True)
            keep = counts > nfolds # Drop everything with fewer labels than folds
            labs = labs[keep]
            counts = counts[keep]
            sel = np.isin(_labels[:, 0], labs)
            _data = _data[sel]
            _labels = _labels[sel]
            _eval_filter_mask = _eval_filter_mask[sel]
            _nclass = len(labs)
            _probs = counts / counts.sum()
            # _baseline_preds = np.random.multinomial(1, _probs, size=len(_labels))
            # _baseline_preds = _baseline_preds.argmax(axis=1)
            # _baseline_preds = labs[_baseline_preds]
            _maj_class = labs[counts.argmax()]
            _maj_class_pred = np.array([_maj_class])
            _baseline_preds = np.tile(_maj_class_pred, [len(_labels)])
            # chance_score = counts.max() / len(_labels)
            # chance_score = f1_score(_labels, _baseline_preds, average='macro')
            chance_score = accuracy_score(_labels, _baseline_preds)
            # _data = convolve(_data, [[[0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25]]], mode='same', method='direct')
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
                stderr('PCA-transforming GloVe components...\n')
                __labels = glove_pca.transform(_labels[:, 1:])
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
                cv_src, labels_src, train_supp_src = partition_cv(_data, _labels, nfolds, eval_filter_mask=_eval_filter_mask)

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
                                _X = np.array_split(_X, math.ceil(len(_X) / nlabperfold), axis=0)
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
                            layer_type='rnn',
                            n_layers=1,
                            n_units=128,
                            kernel_width=1,
                            # cnn_activation=None,
                            n_outputs=306,
                            dropout=None,
                            reg_scale=None,
                            continuous_outputs=True,
                            project=True,
                            batch_normalize=True
                        )
                        X_train = np.transpose(X_train, [0, 2, 1])
                        mu_train = np.transpose(mu_train, [0, 2, 1])
                        denoiser.fit(
                            X_train,
                            X_train - mu_train,
                            batch_size=128,
                            epochs=100,
                            validation_split=0.1,
                            callbacks=[tf.keras.callbacks.EarlyStopping(
                                monitor='val_loss', patience=100, restore_best_weights=True
                            )]
                        )
                        X_train -= denoiser.predict(X_train, batch_size=32)
                        X_train = np.transpose(X_train, [0, 2, 1])

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
                                    _X = np.array_split(_X, math.ceil(len(_X) / nlabperfold), axis=0)
                                    _X = np.stack([__X.mean(axis=0) for __X in _X], axis=0)
                        _y = np.repeat(y_val_src[lab][None, ...], len(_X), axis=0)
                        X_val.append(_X)
                        y_val.append(_y)
                    X_val = np.concatenate(X_val, axis=0)
                    if denoise:
                        X_val = np.transpose(X_val, [0, 2, 1])
                        X_val -= denoiser.predict(X_val, batch_size=128)
                        X_val = np.transpose(X_val, [0, 2, 1])
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
                            # kwargs = {'k_feats': k_feats, 'dropout': dropout}
                            kwargs = {'k_feats': k_feats}
                        elif clstype == 'Ridge':
                            kwargs = {'alpha': 100}
                        elif clstype == 'DNN':
                            kwargs = {
                                'layer_type': layer_type,
                                'learning_rate': learning_rate,
                                'n_units': n_units,
                                'n_layers': n_layers,
                                'kernel_width': kernel_width,
                                'cnn_activation': cnn_activation,
                                'reg_scale': reg_scale,
                                'sensor_filter_scale': sensor_filter_scale,
                                'dropout': dropout,
                                'use_glove': use_glove,
                                'use_resnet': use_resnet,
                                'use_locally_connected': use_locally_connected,
                                'batch_normalize': batch_normalize,
                                'layer_normalize': layer_normalize,
                                'l2_layer_normalize': l2_layer_normalize,
                                'project': project,
                                'contrastive_loss_weight': contrastive_loss_weight,
                            }
                        else:
                            raise ValueError('Unrecognized clstype %s' % clstype)
                        if bytimestep:
                            X_in = X_train[..., t]
                        else:
                            X_in = X_train
                        m = train(
                            X_in,
                            y_train,
                            clstype=clstype,
                            n_dnn_epochs=n_dnn_epochs,
                            dnn_batch_size=dnn_batch_size,
                            **kwargs
                        )

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
                                # s = f1_score(y_val, _y_pred, average='macro')
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
                        plt.axhline(chance_score, color='r')
                    elif score == 'normrank':
                        plt.axhline(0.5, color='r')
                    else:
                        raise ValueError('Unrecognized scoring function: %s' % score)
                    plt.fill_between(x, lb, ub, alpha=0.2)
                    plt.plot(x, mean)
                    plt.xlabel('Time (s)')
                    if score == 'acc':
                        # plt.ylabel('F1')
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


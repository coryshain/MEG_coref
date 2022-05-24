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
    argparser.add_argument('iteration', type=int, help='Which CV iteration to run (1-indexed).')
    argparser.add_argument('fold', type=int, help='Which CV fold to run (1-indexed).')
    argparser.add_argument('-s', '--save_freq', type=int, default=1, help='Save frequency (in epochs).')
    argparser.add_argument('-f', '--force_restart', action='store_true', help='Force model training from initialization, even if a saved model exists.')
    argparser.add_argument('-c', '--cpu_only', action='store_true', help='Force CPU implementation if GPU available.')
    args = argparser.parse_args()

    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=Loader)
    force_restart = args.force_restart
    save_freq = args.save_freq
    if args.cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    paths = config['paths']
    label_field = config.get('label_field', 'attr_cat')
    filters = config.get('filters', {})
    powerband = config.get('powerband', None)
    downsample_by = config.get('downsample_by', 1)
    use_glove = config.get('use_glove', False)
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
    input_dropout = config.get('input_dropout', None)
    use_resnet = config.get('use_resnet', False)
    use_locally_connected = config.get('use_locally_connected', False)
    batch_normalize = config.get('batch_normalize', False)
    layer_normalize = config.get('layer_normalize', False)
    l2_layer_normalize = config.get('l2_layer_normalize', False)
    project = config.get('project', True)
    contrastive_loss_weight = config.get('contrastive_loss_weight', None)
    n_dnn_epochs = config.get('n_dnn_epochs', 1000)
    dnn_batch_size = config.get('dnn_batch_size', 32)
    score = config.get('score', 'acc')
    confint = config.get('confint', 95)
    onset = config.get('onset', 0.5)
    outdir = config.get('outdir', './results')

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if not os.path.normpath(os.path.realpath(config_path)) == os.path.normpath(os.path.realpath(outdir + '/config.ini')):
        shutil.copy2(config_path, outdir + '/config.ini')

    data = []
    labels = []
    ntime = None
    for dirpath in paths:
        dirpath = os.path.normpath(dirpath)
        stderr('Loading %s...\n' % dirpath)
        filename = 'data_d%d_p%s.obj' % (downsample_by, '%s-%s' % tuple(powerband) if powerband else 'None')
        cache_path = os.path.join(dirpath, filename)
        with open(cache_path, 'rb') as f:
            data_src = pickle.load(f)
        _data = data_src['data']
        _labels = data_src['labels']
        meta = data_src['meta']

        _label_df = pd.DataFrame(_labels)
        if 'labcount' not in _label_df.columns:
            labs, ix, counts = np.unique(_label_df[label_field].values, return_inverse=True, return_counts=True)
            counts = counts[ix]
            _label_df['labcount'] = counts
        _filter_mask = compute_filter_mask(_label_df, filters)
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
        _data = np.transpose(_data, [0, 2, 1])
        _labels = _labels[_filter_mask]

        data.append(_data)
        labels.append(_labels)

    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)

    if use_glove and k_pca_glove:
        stderr('Selecting top %d principal GloVe components...\n' % k_pca_glove)
        vocab, ix = np.unique(labels[:, 0], return_index=True)
        _pca_in = labels[ix, 1:]
        glove_pca = Pipeline([('scaler', StandardScaler()), ('pca', PCA(k_pca_glove))])
        glove_pca.fit(_pca_in)

    data = data
    labels = labels
    stderr('Regressing model...\n')
    labs, counts = np.unique(labels[:, 0], return_counts=True)
    _nclass = len(labs)
    _maj_class = labs[counts.argmax()]
    _maj_class_pred = np.array([_maj_class])
    _baseline_preds = np.tile(_maj_class_pred, [len(labels)])
    chance_score_acc = accuracy_score(labels[:, 0], _baseline_preds)
    _probs = counts / counts.sum()
    _baseline_preds = np.random.multinomial(1, _probs, size=len(labels))
    _baseline_preds = _baseline_preds.argmax(axis=1)
    _baseline_preds = labs[_baseline_preds]
    chance_score_f1 = f1_score(labels[:, 0], _baseline_preds, average='macro')
    if k_pca or k_ica:
        _data = data.transpose([0, 2, 1])
        b, t = _data.shape[:2]
        _data = _data.reshape((-1, data.shape[1]))
        if k_pca:
            stderr('Selecting top %d principal components...\n' % k_pca)
            _data = zscore(_data, axis=0)
            pca = PCA(k_pca)
            data = pca.fit_transform(_data)
            data = data.reshape((b, t, k_pca))
            data = data.transpose([0, 2, 1])
            data = data[:, k_pca_drop:, :]
            stderr('  %.2f%% variance explained\n' % (pca.explained_variance_ratio_[k_pca_drop:].sum() * 100))
        if k_ica:
            stderr('Selecting %d independent components...\n' % k_ica)
            data = FastICA(k_ica).fit_transform(_data)
            data = data.reshape((b, t, k_ica))
            data = data.transpose([0, 2, 1])
    if use_glove and k_pca_glove:
        stderr('PCA-transforming GloVe components...\n')
        _labels = glove_pca.transform(labels[:, 1:])
        labels = np.concatenate([labels[:, :1], _labels], axis=1)
    if ntime is None:
        ntime = data.shape[-1]

    i = args.iteration
    j = args.fold
    stderr('Iteration %d, CV fold %d\n' % (i, j))
    fold_path = os.path.join(os.path.normpath(outdir), 'i%d' % i, 'f%d' % j)
    train_ix_path = os.path.join(fold_path, 'train_ix.obj')
    with open(train_ix_path, 'rb') as f:
        train_ix = pickle.load(f)
    val_ix_path = os.path.join(fold_path, 'val_ix.obj')
    with open(val_ix_path, 'rb') as f:
        val_ix = pickle.load(f)

    X_train = data[train_ix]
    y_train_lab = labels[train_ix, 0]
    y_lab_uniq, y_lab_counts = np.unique(y_train_lab, return_counts=True)
    lab_map = {_y: i for i, _y in enumerate(y_lab_uniq)}
    y_train_lab_ix = np.vectorize(lab_map.__getitem__)(y_train_lab)
    X_val = data[val_ix]
    y_val_lab = labels[val_ix, 0]
    if use_glove:
        y_train_glove = labels[train_ix, 1:].astype('float32')
        y_val_glove = labels[val_ix, 1:].astype('float32')
        uniq, ix = np.unique(y_val_lab, return_index=True)
        comparison_set = {}
        for k, val in zip(ix, uniq):
            comparison_set[val] = y_val_glove[k]
        y_train = y_train_glove
        n_outputs = y_train_glove.shape[1]
        monitor = 'val_sim'
    else:
        y_train_glove = None
        y_val_glove = None
        comparison_set = None
        y_train = y_train_lab_ix
        n_outputs = len(lab_map)
        monitor = 'val_acc'

    validation_split = 0.1
    n_train = int(len(X_train) * (1 - validation_split))

    if contrastive_loss_weight:
        ds_train = RasterData(
            X_train[:n_train],
            y=y_train[:n_train],
            batch_size=dnn_batch_size,
            shuffle=True,
            contrastive_sampling=True
        )
        ds_val = RasterData(
            X_train[n_train:],
            y=y_train[n_train:],
            batch_size=dnn_batch_size,
            shuffle=False,
            contrastive_sampling=True
        )
    else:
        ds_train = RasterData(
            X_train[:n_train],
            y=y_train[:n_train],
            batch_size=dnn_batch_size,
            shuffle=True
        )
        ds_val = RasterData(
            X_train[n_train:],
            y=y_train[n_train:],
            batch_size=dnn_batch_size,
            shuffle=False
        )

    ds_test = RasterData(X_val, batch_size=dnn_batch_size, shuffle=False)

    model_path = os.path.join(fold_path, 'm.obj')
    results_path = os.path.join(fold_path, 'results.obj')
    results_dict = {
        'acc': None,
        'f1': None,
        'chance_acc': chance_score_acc,
        'chance_f1': chance_score_f1,
    }
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=100, restore_best_weights=True),
        tf.keras.callbacks.TensorBoard(os.path.join(model_path, 'tensorboard')),
        ModelEval(
            save_freq=save_freq,
            save_path=model_path,
            eval_data_generator=ds_test,
            ylab=y_val_lab,
            comparison_set=comparison_set,
            results_dict=results_dict,
            results_path=results_path
        )
    ]

    m = DNN(
        lab_map=lab_map,
        layer_type=layer_type,
        learning_rate=learning_rate,
        n_units=n_units,
        n_layers=n_layers,
        kernel_width=kernel_width,
        cnn_activation=cnn_activation,
        n_outputs=n_outputs,
        reg_scale=reg_scale,
        sensor_filter_scale=sensor_filter_scale,
        dropout=dropout,
        input_dropout=input_dropout,
        use_glove=use_glove,
        use_resnet=use_resnet,
        use_locally_connected=use_locally_connected,
        batch_normalize=batch_normalize,
        layer_normalize=layer_normalize,
        l2_layer_normalize=l2_layer_normalize,
        project=project,
        contrastive_loss_weight=contrastive_loss_weight
    )
    # Call once to initialize
    m(ds_train[0][0])
    print(m.summary())

    if not force_restart and os.path.exists(model_path):
        stderr('Loading saved checkpoint...\n')
        m = tf.keras.models.load_model(model_path)

    m.fit(
        ds_train,
        epochs=n_dnn_epochs,
        shuffle=False,
        callbacks=callbacks,
        validation_data=ds_val
    )

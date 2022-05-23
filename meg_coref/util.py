import sys
import numpy as np


def stderr(x):
    sys.stderr.write(x)
    sys.stderr.flush()


def partition_cv(data, labels, nfolds, eval_filter_mask=None):
    label_keys = labels[:,0]
    labels_out = {}

    if eval_filter_mask is not None:
        supp_mask = ~eval_filter_mask
        data_supp = data[supp_mask]
        label_keys_supp = label_keys[supp_mask]
        labels_supp = labels[supp_mask]
        data = data[eval_filter_mask]
        label_keys = label_keys[eval_filter_mask]
        labels = labels[eval_filter_mask]

        p = np.random.permutation(np.arange(len(label_keys_supp)))
        data_supp = data_supp[p]
        label_keys_supp = label_keys_supp[p]
        labels_supp = labels_supp[p]
        unique, indices, counts = np.unique(label_keys_supp, return_inverse=True, return_counts=True)
        supp_out = {}

        for i, lab in enumerate(unique):
            ix = np.where(indices == i)
            _data_supp = data_supp[ix]
            labels_out[lab] = labels_supp[ix][0] # All labels in group identical, take first
            supp_out[lab] = _data_supp
    else:
        supp_out = {}

    p = np.random.permutation(np.arange(len(label_keys)))
    data = data[p]
    label_keys = label_keys[p]
    labels = labels[p]
    unique, indices, counts = np.unique(label_keys, return_inverse=True, return_counts=True)
    data_out = [{} for _ in range(nfolds)]

    for i, lab in enumerate(unique):
        ix = np.where(indices == i)
        _data = np.array_split(data[ix], nfolds, axis=0)
        labels_out[lab] = labels[ix][0] # All labels in group identical, take first
        for j, __data in enumerate(_data):
            data_out[j][lab] = __data

    return data_out, labels_out, supp_out


def compute_filter_mask(y, filters):
    for x in y:
        sel = np.ones(len(y[x]), dtype=bool)
        break
    for filter in filters:
        vals = filters[filter]
        sel &= np.isin(y[filter], vals)

    return sel


def normalize(x, axis=-1):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / n
import os
import pickle
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import numpy as np
from scipy.io import loadmat
from scipy.signal import resample
import argparse
import mne

from .util import stderr

MAX_N_RESAMP = 100


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Cache data for loading into decoder.
    ''')
    argparser.add_argument('config', help='Path to config file containing paths to MEG raster data and preprocessing instructions.')
    argparser.add_argument('-f', '--force_reprocess', action='store_true', help='Force data reprocessing, even if a data cache exists.')
    args = argparser.parse_args()

    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=Loader)
    force_reprocess = args.force_reprocess

    paths = config['paths']
    downsample_by = config.get('downsample_by', 1)
    powerband = config.get('powerband', None)
    outdir = config.get('outdir', './results')

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    data = []
    labels = []
    for dirpath in paths:
        dirpath = os.path.normpath(dirpath)
        stderr('Loading %s...\n' % dirpath)
        filename = 'data_d%d_p%s.obj' % (downsample_by, '%s-%s' % tuple(powerband) if powerband else 'None')
        cache_path = os.path.join(dirpath, filename)

        if force_reprocess or not os.path.exists(cache_path):
            raster_data = []
            _labels = None
            raster_paths = [os.path.join(dirpath, x) for x in os.listdir(dirpath) if x.endswith('mat')]
            meta = {}
            for i in range(len(raster_paths)):
                raster_path = raster_paths[i]
                stderr('\r  Sensor %d/%d' % (i + 1, len(raster_paths)))
                raster = loadmat(raster_path, simplify_cells=True)
                _data = raster['raster_data']
                if powerband:
                    meta['powerband'] = powerband
                    l, u = powerband
                    _data = mne.io.RawArray(
                        _data,
                        mne.create_info([str(x) for x in range(len(_data))], 1000, 'grad'),
                        verbose=40
                    )
                    _data = _data.filter(l, u, verbose=40)\
                        .apply_hilbert(envelope=True, verbose=40)\
                        .get_data()
                if downsample_by > 1:
                    meta['downsample_by'] = downsample_by
                    num = _data.shape[1] // downsample_by
                    _data = resample(_data, num, axis=1)
                raster_data.append(_data)
                if _labels is None:
                    _labels = raster['raster_labels']

            _data = np.stack(raster_data, axis=1)

            data_src = {
                'data': _data,
                'labels': _labels,
                'meta': meta
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(data_src, f)

            stderr('\n')

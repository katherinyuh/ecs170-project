"""Check all supplied images, split sizes and class counts without training a model."""

import argparse
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_code.stage_3_code.Dataset_Loader_Images import DATA_DIRECTORY
from local_code.stage_3_code.Dataset_Loader_MNIST import Dataset_Loader_MNIST
from local_code.stage_3_code.Dataset_Loader_ORL import Dataset_Loader_ORL
from local_code.stage_3_code.Dataset_Loader_CIFAR import Dataset_Loader_CIFAR
import numpy as np


LOADERS = {'MNIST': Dataset_Loader_MNIST, 'ORL': Dataset_Loader_ORL,
           'CIFAR': Dataset_Loader_CIFAR}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['all', *LOADERS], default='all')
    parser.add_argument('--data-dir', type=Path, default=DATA_DIRECTORY)
    parser.add_argument('--output', type=Path, help='Optional JSON summary destination')
    args = parser.parse_args()
    names = list(LOADERS) if args.dataset == 'all' else [args.dataset]
    summary = {}
    for name in names:
        loader = LOADERS[name]()
        loader.dataset_source_folder_path = args.data_dir
        data = loader.load()
        summary[name] = {}
        for split in ('train', 'test'):
            X, y = data[split]['X'], data[split]['y']
            if not X.flags.c_contiguous:
                raise ValueError(f'{name} {split}: X must be contiguous.')
            summary[name][split] = {
                'X_shape': list(X.shape), 'y_shape': list(y.shape),
                'X_dtype': str(X.dtype), 'y_dtype': str(y.dtype),
                'pixel_min': float(X.min()), 'pixel_max': float(X.max()),
                'label_counts': np.bincount(y, minlength=loader.num_classes).tolist(),
            }
        # Release each dataset before moving on to the next large file.
        del X, y, data, loader
        gc.collect()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + '\n')
        print(f'Saved summary: {args.output}')
    print('PASS: all requested datasets have the expected shapes, ranges and label counts.')


if __name__ == '__main__':
    main()

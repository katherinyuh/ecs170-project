'''
Smoke test for the Stage 3 CIFAR CNN using small fake data shaped like the
real loader output (N, 3, 32, 32) in [0, 1] with 10 classes, so it runs in
seconds without the data files.
Run from repo root:
  python script/stage_3_script/test_cnn_cifar.py
'''

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR


def make_fake_split(n, rng):
    # each class is a different solid color with noise, so a CNN can learn it fast
    y = rng.integers(0, 10, size=n)
    colors = np.linspace(0.05, 0.95, 10)
    X = np.empty((n, 3, 32, 32), dtype=np.float32)
    for c in range(3):
        X[:, c] = colors[(y + 3 * c) % 10][:, None, None]
    X += rng.normal(0, 0.1, size=X.shape).astype(np.float32)
    return {'X': np.clip(X, 0, 1), 'y': y.astype(np.int64)}


def main():
    rng = np.random.default_rng(2)
    method = Method_CNN_CIFAR('CIFAR CNN', 'smoke test', config={
        'max_epoch': 5, 'batch_size': 64, 'eval_batch_size': 256,
    })
    method.data = {'train': make_fake_split(1000, rng), 'test': make_fake_split(300, rng)}
    result = method.run()

    method.print_shape_table()
    assert len(result['pred_y']) == len(result['true_y']), 'pred_y and true_y lengths differ'
    assert len(method.loss_history) == method.max_epoch, 'loss_history not filled each epoch'
    assert len(method.test_acc_history) == method.max_epoch, 'test_acc_history not filled'
    assert method.loss_history[-1] < method.loss_history[0], 'loss did not go down'

    metrics = Evaluate_Metrics('metrics', 'smoke test')
    metrics.data = result
    scores = metrics.evaluate()
    print('Loss:', round(method.loss_history[0], 4), '->', round(method.loss_history[-1], 4))
    for key, value in scores.items():
        print(f'  {key}: {value:.4f}')
    assert scores['accuracy'] > 0.5, 'model did not learn the fake data'
    print('CIFAR CNN smoke test passed on', method.device)


if __name__ == '__main__':
    from run_log import start_log
    start_log('test_cnn_cifar')
    main()

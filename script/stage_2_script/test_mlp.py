'''
Smoke test for the Stage 2 MLP using fake data shaped like the real dataset
(784 pixel values from 0-255, 10 classes), so it runs without the data files.
Run from repo root:
  PYTHONPATH=. python script/stage_2_script/test_mlp.py
'''

import numpy as np
import torch
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics


def make_fake_split(n, centers, rng):
    y = rng.integers(0, 10, size=n)
    X = np.clip(centers[y] + rng.normal(0, 40, size=(n, 784)), 0, 255).astype(int)
    return {'X': X.tolist(), 'y': y.tolist()}


def main():
    np.random.seed(2)
    torch.manual_seed(2)
    rng = np.random.default_rng(2)
    # each class gets its own random "average image" so the model has something to learn
    centers = rng.integers(0, 256, size=(10, 784))

    method = Method_MLP('multi-layer perceptron', 'smoke test')
    method.max_epoch = 30
    method.data = {
        'train': make_fake_split(2000, centers, rng),
        'test': make_fake_split(500, centers, rng),
    }
    result = method.run()

    assert len(result['pred_y']) == len(result['true_y']), 'pred_y and true_y lengths differ'
    assert len(method.loss_history) == method.max_epoch, 'loss_history not filled each epoch'
    assert method.loss_history[-1] < method.loss_history[0], 'loss did not go down'

    metrics = Evaluate_Metrics('metrics', 'smoke test')
    metrics.data = result
    print('Loss:', round(method.loss_history[0], 4), '->', round(method.loss_history[-1], 4))
    for key, value in metrics.evaluate().items():
        print(f'  {key}: {value:.4f}')
    print('MLP smoke test passed')


if __name__ == '__main__':
    main()

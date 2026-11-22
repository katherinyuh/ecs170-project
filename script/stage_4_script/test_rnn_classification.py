'''
Smoke test for the Stage 4 classification model (RNN, LSTM and GRU) using small fake data
shaped like Part A's loader output: (N, 200) word IDs, pre-padded with 0, labels 0/1.
Runs in under a minute without the data files.
Run from repo root (or press Run in PyCharm):
  python script/stage_4_script/test_rnn_classification.py
'''

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_4_code.Method_RNN_Classification import Method_RNN_Classification

VOCAB, MAX_LEN = 500, 200
POS_WORDS, NEG_WORDS = range(10, 20), range(20, 30)


def make_fake_split(n, rng):
    '''Reviews of random length; positive ones contain "good" words, negative ones "bad" words.'''
    X = np.zeros((n, MAX_LEN), dtype=np.int64)
    y = rng.integers(0, 2, size=n)
    for i in range(n):
        length = rng.integers(20, MAX_LEN + 1)
        words = rng.integers(30, VOCAB, size=length)
        signal = list(POS_WORDS if y[i] else NEG_WORDS)
        k = max(4, length // 5)          # about 1 in 5 words carries the sentiment
        spots = rng.choice(length, size=k, replace=False)
        words[spots] = rng.choice(signal, size=k)
        X[i, -length:] = words          # pre-padding, like the real loader
    return {'X': X, 'y': y}


def main():
    rng = np.random.default_rng(2)
    train, test = make_fake_split(1500, rng), make_fake_split(400, rng)
    for cell in ('rnn', 'lstm', 'gru'):
        m = Method_RNN_Classification(cell, 'smoke test', config={
            'cell_type': cell, 'max_epoch': 5, 'hidden_size': 64, 'embedding_dim': 32})
        m.vocab_size = VOCAB
        m.data = {'train': train, 'test': test}
        result = m.run()
        assert len(result['pred_y']) == len(result['true_y'])
        assert len(m.loss_history) == m.max_epoch and len(m.test_acc_history) == m.max_epoch
        assert m.loss_history[-1] < m.loss_history[0], f'{cell}: loss did not go down'
        metrics = Evaluate_Metrics('metrics', '')
        metrics.data = result
        acc = metrics.evaluate()['accuracy']
        print(f'{cell.upper()}: loss {m.loss_history[0]:.3f} -> {m.loss_history[-1]:.3f}, test accuracy {acc:.3f}')
        assert acc > 0.75, f'{cell}: did not learn the fake data'
    m.print_parameter_table()
    print('RNN classification smoke test passed on', m.device)


if __name__ == '__main__':
    from run_log import start_log
    start_log('test_rnn_classification')
    main()

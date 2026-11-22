'''
Person B: Stage 4 text classification (IMDB sentiment) with RNN, LSTM and GRU.
Trains each cell type with the same settings, saves a learning curve for each, a results
table (CSV), a comparison bar chart and a log.

Run from the repo root (or press Run in PyCharm):
  python script/stage_4_script/run_classification_experiments.py            # rnn, lstm, gru
  python script/stage_4_script/run_classification_experiments.py --only lstm
  python script/stage_4_script/run_classification_experiments.py --quick    # 1 epoch on 5,000 reviews, to check it works
  python script/stage_4_script/run_classification_experiments.py --epochs 5
  python script/stage_4_script/run_classification_experiments.py --no-glove   # learn embeddings from scratch

GloVe: if a 100d GloVe file (txt or zip) is anywhere in data/stage_4_data/, it is used
automatically to start the embedding layer (fine-tuned during training).

Each finished run is written to CLS_experiment_results.csv right away, so you can stop
and continue later with --only.
'''

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_code.base_class.dataset import dataset
from local_code.stage_1_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_2_code.Setting_Train_Test_PreSplit import Setting_Train_Test_PreSplit
from local_code.stage_4_code.Dataset_Loader_Classification import Dataset_Loader_Classification
from local_code.stage_4_code.Method_RNN_Classification import Method_RNN_Classification
from local_code.stage_4_code.Glove_Utils import find_glove_file, load_glove_matrix

RESULT_DIR = ROOT / 'result' / 'stage_4_result' / 'classification'

# training settings shared by every run, so the only difference is the cell type
SHARED = {
    'embedding_dim': 100,
    'hidden_size': 128,
    'num_layers': 1,
    'bidirectional': False,
    'dropout': 0.3,
    'max_epoch': 10,
    'learning_rate': 1e-3,
    'batch_size': 64,
    'grad_clip': 5.0,
    'optimizer_name': 'adam',
    'seed': 2,
}

EXPERIMENTS = [
    {'name': 'rnn', 'config': {'cell_type': 'rnn'}},
    {'name': 'lstm', 'config': {'cell_type': 'lstm'}},
    {'name': 'gru', 'config': {'cell_type': 'gru'}},
]

METRIC_KEYS = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
               'precision_weighted', 'recall_weighted', 'f1_weighted']


class Cached_Dataset(dataset):
    '''Hands the already-loaded data to the Setting so 50,000 reviews are read only once.'''

    def __init__(self, data):
        super().__init__('IMDB', 'movie review sentiment (cached)')
        self.data = data

    def load(self):
        return self.data


def save_learning_curves(m, out_path, title):
    epochs = np.arange(1, len(m.loss_history) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, m.loss_history, marker='o', label='train loss')
    axes[0].set(title='Training Loss', xlabel='Epoch', ylabel='Cross-entropy loss')
    axes[0].legend()
    axes[1].plot(epochs, m.train_acc_history, marker='o', label='train acc')
    if m.test_acc_history:
        axes[1].plot(epochs, m.test_acc_history, marker='o', label='test acc')
    axes[1].set(title='Accuracy', xlabel='Epoch', ylabel='Accuracy', ylim=(0.4, 1.02))
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.set_xticks(epochs)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_comparison_plot(csv_path, out_path):
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    order = {'rnn': 0, 'lstm': 1, 'gru': 2}
    rows.sort(key=lambda r: order.get(r['name'], 9))
    names = [r['name'].upper() for r in rows]
    accs = [float(r['accuracy']) for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(names, accs, color='#4C72B0', width=0.55)
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.005, f'{a:.3f}', ha='center', va='bottom', fontsize=10)
    ax.set(title='IMDB sentiment: test accuracy by recurrent cell', ylabel='Test accuracy',
           ylim=(max(0.0, min(accs) - 0.1), min(1.0, max(accs) + 0.05)))
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def append_row(csv_path, row):
    rows = []
    if csv_path.exists():
        with csv_path.open() as f:
            rows = [r for r in csv.DictReader(f) if r['name'] != row['name']]
    rows.append({k: str(v) for k, v in row.items()})
    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerows(rows)


def run_one(exp, data, vocab_size, result_dir, overrides, glove=None):
    name = exp['name']
    config = {**SHARED, **exp['config'], **overrides}
    print('\n' + '=' * 60)
    print(f"Run: {name}  (cell_type={config['cell_type']})")
    print('=' * 60, flush=True)

    method_obj = Method_RNN_Classification(f'IMDB {name.upper()}', '', config=config)
    method_obj.vocab_size = vocab_size
    if glove is not None:
        method_obj.embedding_matrix = glove[0]

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = str(result_dir) + os.sep + f'CLS_{name}_'
    result_obj.result_destination_file_name = 'prediction_result'
    result_obj.fold_count = None

    setting_obj = Setting_Train_Test_PreSplit('pre-split train/test', '')
    evaluate_obj = Evaluate_Metrics('metrics', '')
    setting_obj.prepare(Cached_Dataset(data), method_obj, result_obj, evaluate_obj)

    start = time.time()
    scores = setting_obj.load_run_save_evaluate()
    minutes = (time.time() - start) / 60

    print('************ Test Performance ************')
    for k in METRIC_KEYS:
        print(f'{k}: {scores[k]:.4f}')
    print(f'run time: {minutes:.1f} min')
    method_obj.print_parameter_table()

    curve = result_dir / f'CLS_learning_curve_{name}.png'
    save_learning_curves(method_obj, curve, f'IMDB sentiment, {name.upper()}')
    print('Saved learning curve:', curve)

    row = {
        'name': name,
        'cell_type': config['cell_type'],
        'embedding_dim': config['embedding_dim'],
        'hidden_size': config['hidden_size'],
        'num_layers': config['num_layers'],
        'bidirectional': config['bidirectional'],
        'dropout': config['dropout'],
        'max_epoch': config['max_epoch'],
        'learning_rate': config['learning_rate'],
        'batch_size': config['batch_size'],
        'grad_clip': config['grad_clip'],
        'embedding': (f"GloVe {glove[1]['glove_dim']}d" if glove is not None else 'learned from scratch'),
        'freeze_embeddings': config.get('freeze_embeddings', False) if glove is not None else '',
        'max_len': data['max_len'] if 'max_len' in data else '',
        'vocab_size': vocab_size,
        'train_reviews': config.get('train_subset') or len(data['train']['y']),
        'params': sum(r[2] for r in method_obj.parameter_table()),
        'device': str(method_obj.device),
        'minutes': round(minutes, 2),
        'final_train_loss': method_obj.loss_history[-1],
        'final_train_acc': method_obj.train_acc_history[-1],
        'best_test_acc_curve': max(method_obj.test_acc_history) if method_obj.test_acc_history else '',
        'learning_curve': curve.name,
    }
    for k in METRIC_KEYS:
        row[k] = scores[k]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--only', nargs='+', choices=[e['name'] for e in EXPERIMENTS])
    parser.add_argument('--epochs', type=int, help='override max_epoch for every run')
    parser.add_argument('--quick', action='store_true',
                        help='1 epoch on 5,000 training reviews, results go to classification_quick/')
    parser.add_argument('--cpu', action='store_true', help='force CPU even if a GPU exists')
    parser.add_argument('--glove', help='path to a GloVe .txt or .zip (default: looks in data/stage_4_data/)')
    parser.add_argument('--no-glove', action='store_true', help='learn the embedding from scratch instead')
    parser.add_argument('--freeze', action='store_true', help='keep GloVe vectors fixed instead of fine-tuning')
    args = parser.parse_args()

    from run_log import start_log
    start_log('run_classification_experiments' + ('_quick' if args.quick else ''))

    overrides, result_dir = {}, RESULT_DIR
    if args.quick:
        overrides.update({'max_epoch': 1, 'train_subset': 5000})
        result_dir = RESULT_DIR.parent / 'classification_quick'
    if args.epochs:
        overrides['max_epoch'] = args.epochs
    if args.cpu:
        overrides['device_name'] = 'cpu'
    result_dir.mkdir(parents=True, exist_ok=True)

    print('************ Start IMDB classification experiments ************')
    loader = Dataset_Loader_Classification()
    data = loader.load()
    vocab_size = loader.stats['vocab_size']
    print(f"train X {data['train']['X'].shape}, test X {data['test']['X'].shape}, vocab {vocab_size}, "
          f"max_len {data['max_len']}", flush=True)

    glove = None
    if not args.no_glove:
        glove_path = Path(args.glove) if args.glove else find_glove_file(SHARED['embedding_dim'])
        if glove_path is None:
            print('No GloVe file found in data/stage_4_data/, so the embedding is learned from scratch.')
        else:
            print('Loading GloVe vectors from', glove_path, flush=True)
            glove = load_glove_matrix(glove_path, data['vocab']['id_to_word'])
            g = glove[1]
            print(f"GloVe {g['glove_dim']}d: {g['found_in_glove']} of {g['vocab_words']} vocab words found "
                  f"({g['coverage']:.1%}), {g['seconds']}s", flush=True)
            if args.freeze:
                overrides['freeze_embeddings'] = True

    experiments = [e for e in EXPERIMENTS if not args.only or e['name'] in args.only]
    csv_path = result_dir / 'CLS_experiment_results.csv'
    for exp in experiments:
        append_row(csv_path, run_one(exp, data, vocab_size, result_dir, overrides, glove))
        print('Updated', csv_path, flush=True)

    save_comparison_plot(csv_path, result_dir / 'CLS_accuracy_comparison.png')
    print('\n************ Results table ************')
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    print(f"{'cell':<6} {'acc':>7} {'f1_macro':>9} {'precision':>10} {'recall':>8} {'min':>6}")
    for r in rows:
        print(f"{r['name']:<6} {float(r['accuracy']):7.4f} {float(r['f1_macro']):9.4f} "
              f"{float(r['precision_macro']):10.4f} {float(r['recall_macro']):8.4f} {float(r['minutes']):6.1f}")
    print('************ Finish ************')


if __name__ == '__main__':
    main()

'''
Person B: MNIST CNN experiments (baseline + one-change-at-a-time variations).
Same layout as run_orl_experiments.py / run_cifar_experiments.py so the report tables line up.

Run from the repo root (or press Run in PyCharm):
  python script/stage_3_script/run_mnist_experiments.py                  # baseline + all variations
  python script/stage_3_script/run_mnist_experiments.py --only baseline  # just the full baseline run
  python script/stage_3_script/run_mnist_experiments.py --epochs 2       # quick check that everything works

Each finished run is appended to MNIST_experiment_results.csv right away,
so you can stop and continue later with --only.
'''

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_code.stage_1_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_3_code.Dataset_Loader_MNIST import Dataset_Loader_MNIST
from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST
from local_code.stage_3_code.Setting_Train_Test_PreSplit import Setting_Train_Test_PreSplit


RESULT_DIR = ROOT / 'result' / 'stage_3_result' / 'MNIST'
# shared training knobs for every run (architecture varies below)
SHARED = {
    'max_epoch': 10,
    'learning_rate': 1e-3,
    'batch_size': 128,
    'optimizer_name': 'adam',
    'print_every': 1,
    'seed': 2,
    'track_test_acc': True,
}

BASELINE = {
    'conv_channels': [32, 64],
    'kernel_size': 3,
    'padding': None,   # None = "same" padding (kernel_size // 2)
    'stride': 1,
    'pool_type': 'max',
    'fc_hidden': 128,
    'dropout': 0.25,
    'loss_name': 'cross_entropy',
}

# baseline first; each later run changes exactly one field named in "change"
VARIATIONS = [
    ('depth1', 'conv_channels', {'conv_channels': [32]}),
    ('depth3', 'conv_channels', {'conv_channels': [32, 64, 128]}),
    ('kernel5', 'kernel_size', {'kernel_size': 5}),
    ('padding0', 'padding', {'padding': 0}),
    ('stride2', 'stride', {'stride': 2}),
    ('avgpool', 'pool_type', {'pool_type': 'avg'}),
    ('nopool', 'pool_type', {'pool_type': 'none'}),
    ('fc256', 'fc_hidden', {'fc_hidden': 256}),
    ('multimargin', 'loss_name', {'loss_name': 'multi_margin'}),
]
EXPERIMENTS = [{'name': 'baseline', 'change': 'none', 'config': dict(BASELINE)}] + [
    {'name': name, 'change': change, 'config': {**BASELINE, **override}}
    for name, change, override in VARIATIONS
]

# metrics written to the CSV (Evaluate_Metrics returns more; these are the report ones)
METRIC_KEYS = [
    'accuracy',
    'precision_macro',
    'precision_weighted',
    'recall_macro',
    'recall_weighted',
    'f1_macro',
    'f1_weighted',
]


def config_tag(config):
    """Short filename-safe summary of the architecture knobs."""
    ch = '-'.join(str(c) for c in config['conv_channels'])
    return (
        f"ch{ch}_k{config['kernel_size']}_p{config['padding']}_s{config['stride']}_{config['pool_type']}_"
        f"fc{config['fc_hidden']}_do{config['dropout']}_{config['loss_name']}"
    )


def save_learning_curves(method_obj, out_path, title):
    epochs = np.arange(1, len(method_obj.loss_history) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, method_obj.loss_history, label='train loss')
    axes[0].set(title='Training Loss', xlabel='Epoch', ylabel='Loss')
    axes[0].legend()

    axes[1].plot(epochs, method_obj.train_acc_history, label='train acc')
    if method_obj.test_acc_history:
        axes[1].plot(epochs, method_obj.test_acc_history, label='test acc')
    axes[1].set(title='Accuracy', xlabel='Epoch', ylabel='Accuracy')
    axes[1].legend()

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_one(exp, result_dir, epochs=None):
    name = exp['name']
    config = {**SHARED, **exp['config']}
    if epochs is not None:
        config['max_epoch'] = epochs
    tag = config_tag(exp['config'])
    print('\n' + '=' * 60)
    print(f"Run: {name}  (change={exp['change']})  [{tag}]")
    print('=' * 60)

    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    data_obj = Dataset_Loader_MNIST('MNIST', 'Handwritten digits')
    method_obj = Method_CNN_MNIST(f'MNIST CNN {name}', '', config=config)

    result_obj = Result_Saver('saver', '')
    # Result_Saver concatenates folder + file + '_' + fold_count
    result_obj.result_destination_folder_path = str(result_dir) + os.sep + f'MNIST_{name}_'
    result_obj.result_destination_file_name = 'prediction_result'
    result_obj.fold_count = None

    setting_obj = Setting_Train_Test_PreSplit('pre-split train/test', '')
    evaluate_obj = Evaluate_Metrics('multiclass metrics', '')

    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    scores = setting_obj.load_run_save_evaluate()

    print('************ Test Performance ************')
    for key in METRIC_KEYS:
        print(f'{key}: {scores[key]:.4f}')

    curve_path = result_dir / f'MNIST_learning_curve_{name}_{tag}.png'
    save_learning_curves(method_obj, curve_path, title=f'MNIST CNN ({name})')
    print(f'Saved learning curve: {curve_path}')

    if name == 'baseline':
        print('\n************ Baseline layer shapes ************')
        method_obj.print_shape_table()

    row = {
        'name': name,
        'change': exp['change'],
        'conv_channels': str(exp['config']['conv_channels']),
        'kernel_size': exp['config']['kernel_size'],
        'padding': exp['config']['padding'],
        'stride': exp['config']['stride'],
        'pool_type': exp['config']['pool_type'],
        'fc_hidden': exp['config']['fc_hidden'],
        'dropout': exp['config']['dropout'],
        'loss_name': exp['config']['loss_name'],
        'max_epoch': config['max_epoch'],
        'learning_rate': config['learning_rate'],
        'batch_size': config['batch_size'],
        'final_train_loss': method_obj.loss_history[-1],
        'final_train_acc': method_obj.train_acc_history[-1],
        'final_test_acc_curve': method_obj.test_acc_history[-1] if method_obj.test_acc_history else '',
        'learning_curve': curve_path.name,
    }
    for key in METRIC_KEYS:
        row[key] = scores[key]
    return row


def append_row(csv_path, row):
    new_file = not csv_path.exists()
    with csv_path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', nargs='+', help='run only these experiment names')
    parser.add_argument('--epochs', type=int, help='override max_epoch (e.g. 2 for a quick check)')
    args = parser.parse_args()

    experiments = EXPERIMENTS
    if args.only:
        unknown = set(args.only) - {e['name'] for e in EXPERIMENTS}
        if unknown:
            parser.error(f'unknown experiment(s): {sorted(unknown)}')
        experiments = [e for e in EXPERIMENTS if e['name'] in args.only]

    result_dir = RESULT_DIR
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = result_dir / 'MNIST_experiment_results.csv'

    print('************ Start MNIST CNN experiments ************')
    print(f'Results -> {result_dir}')

    rows = []
    for exp in experiments:
        row = run_one(exp, result_dir, epochs=args.epochs)
        append_row(csv_path, row)
        rows.append(row)

    print('\n************ Results table ************')
    print(f'Appended to {csv_path}')
    print(f"{'name':<12} {'change':<14} {'acc':>7} {'f1_macro':>9} {'f1_wt':>7}")
    for row in rows:
        print(f"{row['name']:<12} {row['change']:<14} "
              f"{row['accuracy']:7.4f} {row['f1_macro']:9.4f} {row['f1_weighted']:7.4f}")
    print('************ Finish ************')


if __name__ == '__main__':
    main()

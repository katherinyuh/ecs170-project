'''
Person D: CIFAR-10 CNN experiments (baseline + one-change-at-a-time variations).
Same layout as run_orl_experiments.py so the report tables line up.

Run from the repo root (or press Run in PyCharm):
  python script/stage_3_script/run_cifar_experiments.py              # baseline + 4 variations
  python script/stage_3_script/run_cifar_experiments.py --all        # also the 2 extra runs
  python script/stage_3_script/run_cifar_experiments.py --only baseline
  python script/stage_3_script/run_cifar_experiments.py --quick      # 2 epochs on 5,000 images to check it works
  python script/stage_3_script/run_cifar_experiments.py --epochs 10  # fewer epochs per run

Each finished run is appended to CIFAR_experiment_results.csv right away,
so you can stop and continue later with --only.
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
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_code.stage_1_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_3_code.Dataset_Loader_CIFAR import Dataset_Loader_CIFAR
from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
from local_code.stage_3_code.Setting_Train_Test_PreSplit import Setting_Train_Test_PreSplit


RESULT_DIR = ROOT / 'result' / 'stage_3_result' / 'CIFAR'
# shared training knobs for every run (architecture varies below)
SHARED = {
    'max_epoch': 20,
    'learning_rate': 1e-3,
    'batch_size': 128,
    'weight_decay': 5e-4,
    'optimizer_name': 'adam',
    'loss_name': 'cross_entropy',
    'augment': True,
    'print_every': 1,
    'seed': 2,
    'track_test_acc': True,
}

BASELINE = {
    'conv_channels': [32, 64, 128],
    'convs_per_block': 2,
    'kernel_size': 3,
    'pool_type': 'max',
    'use_batchnorm': True,
    'fc_hidden': 256,
    'dropout': 0.5,
}

# baseline first; each later run changes exactly one field named in "change".
# The first five run by default (baseline + 4 variations). The extras only run
# when named, e.g. --only fc512 label_smoothing, or with --all.
EXPERIMENTS = [
    {'name': 'baseline', 'change': 'none', 'config': {}},
    {'name': 'depth2', 'change': 'conv_channels', 'config': {'conv_channels': [32, 64]}},
    {'name': 'kernel5', 'change': 'kernel_size', 'config': {'kernel_size': 5}},
    {'name': 'avgpool', 'change': 'pool_type', 'config': {'pool_type': 'avg'}},
    {'name': 'no_batchnorm', 'change': 'use_batchnorm', 'config': {'use_batchnorm': False}},
    {'name': 'fc512', 'change': 'fc_hidden', 'config': {'fc_hidden': 512}, 'extra': True},
    {'name': 'label_smoothing', 'change': 'loss_name',
     'config': {'loss_name': 'label_smoothing'}, 'extra': True},
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
    bn = 'bn' if config['use_batchnorm'] else 'nobn'
    return (
        f"ch{ch}x{config['convs_per_block']}_k{config['kernel_size']}_"
        f"{config['pool_type']}_{bn}_fc{config['fc_hidden']}_do{config['dropout']}"
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
    axes[1].set(title='Accuracy', xlabel='Epoch', ylabel='Accuracy', ylim=(0, 1.05))
    axes[1].legend()

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_comparison_plot(csv_path, out_path):
    """Bar chart of test accuracy for every finished run, for the report."""
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    names = [r['name'] for r in rows]
    accs = [float(r['accuracy']) for r in rows]
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 4))
    bars = ax.bar(names, accs, color='#4C72B0')
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.005, f'{acc:.3f}',
                ha='center', va='bottom', fontsize=9)
    ax.set(title='CIFAR CNN test accuracy by configuration', ylabel='Test accuracy',
           ylim=(min(accs) - 0.05 if min(accs) > 0.05 else 0, min(1.0, max(accs) + 0.05)))
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_one(exp, result_dir, overrides):
    name = exp['name']
    arch = {**BASELINE, **exp['config']}
    config = {**SHARED, **arch, **overrides}
    tag = config_tag(arch)
    print('\n' + '=' * 60)
    print(f"Run: {name}  (change={exp['change']})  [{tag}]")
    print('=' * 60, flush=True)

    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    data_obj = Dataset_Loader_CIFAR('CIFAR', 'Colored objects')
    method_obj = Method_CNN_CIFAR(f'CIFAR CNN {name}', '', config=config)

    result_obj = Result_Saver('saver', '')
    # Result_Saver concatenates folder + file + '_' + fold_count
    result_obj.result_destination_folder_path = str(result_dir) + os.sep + f'CIFAR_{name}_'
    result_obj.result_destination_file_name = 'prediction_result'
    result_obj.fold_count = None

    setting_obj = Setting_Train_Test_PreSplit('pre-split train/test', '')
    evaluate_obj = Evaluate_Metrics('multiclass metrics', '')

    start = time.time()
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    scores = setting_obj.load_run_save_evaluate()
    minutes = (time.time() - start) / 60

    print('************ Test Performance ************')
    for key in METRIC_KEYS:
        print(f'{key}: {scores[key]:.4f}')
    print(f'run time: {minutes:.1f} min')

    curve_path = result_dir / f'CIFAR_learning_curve_{name}_{tag}.png'
    save_learning_curves(method_obj, curve_path, title=f'CIFAR CNN ({name})')
    print(f'Saved learning curve: {curve_path}')

    if name == 'baseline':
        print('\n************ Baseline layer shapes ************')
        method_obj.print_shape_table()

    row = {
        'name': name,
        'change': exp['change'],
        'conv_channels': str(arch['conv_channels']),
        'convs_per_block': arch['convs_per_block'],
        'kernel_size': arch['kernel_size'],
        'pool_type': arch['pool_type'],
        'use_batchnorm': arch['use_batchnorm'],
        'fc_hidden': arch['fc_hidden'],
        'dropout': arch['dropout'],
        'loss_name': config['loss_name'],
        'augment': config['augment'],
        'max_epoch': config['max_epoch'],
        'learning_rate': config['learning_rate'],
        'weight_decay': config['weight_decay'],
        'batch_size': config['batch_size'],
        'train_subset': config.get('train_subset') or 'all',
        'device': str(method_obj.device),
        'minutes': round(minutes, 2),
        'final_train_loss': method_obj.loss_history[-1] if method_obj.loss_history else '',
        'final_train_acc': method_obj.train_acc_history[-1] if method_obj.train_acc_history else '',
        'final_test_acc_curve': (
            method_obj.test_acc_history[-1] if method_obj.test_acc_history else ''
        ),
        'learning_curve': str(curve_path.name),
    }
    for key in METRIC_KEYS:
        row[key] = scores[key]
    return row


def append_row(csv_path, row):
    """Add one run to the CSV, replacing an older row with the same name."""
    rows = []
    if csv_path.exists():
        with csv_path.open() as f:
            rows = [r for r in csv.DictReader(f) if r['name'] != row['name']]
    rows.append({k: str(v) for k, v in row.items()})
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--only', nargs='+', choices=[e['name'] for e in EXPERIMENTS],
                        help='run only these experiments')
    parser.add_argument('--all', action='store_true', help='also run the extra experiments')
    parser.add_argument('--epochs', type=int, help='override max_epoch for every run')
    parser.add_argument('--quick', action='store_true',
                        help='2 epochs on 5,000 training images, results go to CIFAR_quick/')
    parser.add_argument('--cpu', action='store_true', help='force CPU even if MPS/CUDA exists')
    args = parser.parse_args()

    from run_log import start_log
    start_log('run_cifar_experiments' + ('_quick' if args.quick else ''))

    overrides = {}
    result_dir = RESULT_DIR
    if args.quick:
        overrides.update({'max_epoch': 2, 'train_subset': 5000})
        result_dir = RESULT_DIR.parent / 'CIFAR_quick'
    if args.epochs:
        overrides['max_epoch'] = args.epochs
    if args.cpu:
        overrides['device_name'] = 'cpu'
    result_dir.mkdir(parents=True, exist_ok=True)

    if args.only:
        experiments = [e for e in EXPERIMENTS if e['name'] in args.only]
    else:
        experiments = [e for e in EXPERIMENTS if args.all or not e.get('extra')]
    csv_path = result_dir / 'CIFAR_experiment_results.csv'

    print('************ Start CIFAR CNN experiments ************')
    print(f'Results -> {result_dir}')
    print('Runs:', ', '.join(e['name'] for e in experiments), flush=True)

    for exp in experiments:
        row = run_one(exp, result_dir, overrides)
        append_row(csv_path, row)
        print(f'Updated {csv_path}', flush=True)

    save_comparison_plot(csv_path, result_dir / 'CIFAR_accuracy_comparison.png')

    print('\n************ Results table ************')
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    print(f"{'name':<16} {'change':<15} {'acc':>7} {'f1_macro':>9} {'f1_wt':>7} {'min':>6}")
    for row in rows:
        print(
            f"{row['name']:<16} {row['change']:<15} "
            f"{float(row['accuracy']):7.4f} {float(row['f1_macro']):9.4f} "
            f"{float(row['f1_weighted']):7.4f} {float(row['minutes']):6.1f}"
        )
    print('************ Finish ************')


if __name__ == '__main__':
    main()

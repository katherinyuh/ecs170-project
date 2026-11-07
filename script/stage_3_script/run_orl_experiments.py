'''
Person C: ORL CNN experiments (baseline + one-change-at-a-time variations).

Run from the repo root:
  python script/stage_3_script/run_orl_experiments.py
'''

import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_code.stage_1_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_3_code.Dataset_Loader_ORL import Dataset_Loader_ORL
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from local_code.stage_3_code.Setting_Train_Test_PreSplit import Setting_Train_Test_PreSplit


RESULT_DIR = ROOT / 'result' / 'stage_3_result' / 'ORL'
# shared training knobs for every run (architecture varies below)
SHARED = {
    'max_epoch': 50,
    'learning_rate': 1e-3,
    'batch_size': 32,
    'weight_decay': 1e-4,
    'optimizer_name': 'adam',
    'loss_name': 'cross_entropy',
    'print_every': 5,
    'seed': 2,
    'track_test_acc': True,
}

# baseline first; each later run changes exactly one field named in "change"
EXPERIMENTS = [
    {
        'name': 'baseline',
        'change': 'none',
        'config': {
            'conv_channels': [16, 32],
            'kernel_size': 3,
            'pool_type': 'max',
            'fc_hidden': 128,
            'dropout': 0.5,
        },
    },
    {
        'name': 'depth3',
        'change': 'conv_channels',
        'config': {
            'conv_channels': [16, 32, 64],
            'kernel_size': 3,
            'pool_type': 'max',
            'fc_hidden': 128,
            'dropout': 0.5,
        },
    },
    {
        'name': 'kernel5',
        'change': 'kernel_size',
        'config': {
            'conv_channels': [16, 32],
            'kernel_size': 5,
            'pool_type': 'max',
            'fc_hidden': 128,
            'dropout': 0.5,
        },
    },
    {
        'name': 'avgpool',
        'change': 'pool_type',
        'config': {
            'conv_channels': [16, 32],
            'kernel_size': 3,
            'pool_type': 'avg',
            'fc_hidden': 128,
            'dropout': 0.5,
        },
    },
    {
        'name': 'dropout03',
        'change': 'dropout',
        'config': {
            'conv_channels': [16, 32],
            'kernel_size': 3,
            'pool_type': 'max',
            'fc_hidden': 128,
            'dropout': 0.3,
        },
    },
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
        f"ch{ch}_k{config['kernel_size']}_"
        f"{config['pool_type']}_fc{config['fc_hidden']}_do{config['dropout']}"
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


def run_one(exp, result_dir):
    name = exp['name']
    config = {**SHARED, **exp['config']}
    tag = config_tag(exp['config'])
    print('\n' + '=' * 60)
    print(f"Run: {name}  (change={exp['change']})  [{tag}]")
    print('=' * 60)

    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    data_obj = Dataset_Loader_ORL('ORL', 'Face recognition')
    method_obj = Method_CNN_ORL(f'ORL CNN {name}', '', config=config)

    result_obj = Result_Saver('saver', '')
    # Result_Saver concatenates folder + file + '_' + fold_count
    result_obj.result_destination_folder_path = str(result_dir) + os.sep + f'ORL_{name}_'
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

    curve_path = result_dir / f'ORL_learning_curve_{name}_{tag}.png'
    save_learning_curves(method_obj, curve_path, title=f'ORL CNN ({name})')
    print(f'Saved learning curve: {curve_path}')

    if name == 'baseline':
        print('\n************ Baseline layer shapes ************')
        method_obj.print_shape_table()

    row = {
        'name': name,
        'change': exp['change'],
        'conv_channels': str(exp['config']['conv_channels']),
        'kernel_size': exp['config']['kernel_size'],
        'pool_type': exp['config']['pool_type'],
        'fc_hidden': exp['config']['fc_hidden'],
        'dropout': exp['config']['dropout'],
        'max_epoch': config['max_epoch'],
        'learning_rate': config['learning_rate'],
        'weight_decay': config['weight_decay'],
        'batch_size': config['batch_size'],
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


def main():
    result_dir = RESULT_DIR
    result_dir.mkdir(parents=True, exist_ok=True)

    print('************ Start ORL CNN experiments ************')
    print(f'Results -> {result_dir}')

    rows = [run_one(exp, result_dir) for exp in EXPERIMENTS]

    csv_path = result_dir / 'ORL_experiment_results.csv'
    fieldnames = list(rows[0].keys())
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print('\n************ Results table ************')
    print(f'Wrote {csv_path}')
    header = f"{'name':<12} {'change':<14} {'acc':>7} {'f1_macro':>9} {'f1_wt':>7}"
    print(header)
    for row in rows:
        print(
            f"{row['name']:<12} {row['change']:<14} "
            f"{row['accuracy']:7.4f} {row['f1_macro']:9.4f} {row['f1_weighted']:7.4f}"
        )
    print('************ Finish ************')


if __name__ == '__main__':
    main()

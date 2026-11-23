"""Part D: change one factor at a time; save real metrics, curves and stories.

From repository root:
  python script/stage_4_script/run_part_d.py --task generation
  python script/stage_4_script/run_part_d.py --task classification --epochs 3
  python script/stage_4_script/run_part_d.py --task both --quick
  python script/stage_4_script/run_part_d.py --plan

Classification embeddings are learned unless --glove is supplied. A pretrained
embedding dimension sweep requires separate matching GloVe files, so the runner
refuses that sweep with --glove rather than silently changing initialization.
"""
import argparse
import csv
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
VARIANTS = {
    'baseline': {},
    'embedding50': {'embedding_dim': 50},
    'hidden256': {'hidden_size': 256},
    'layers2': {'num_layers': 2},
    'sequence': {},  # generation context 3 -> 5; classification review cap 200 -> 300
    'lr0005': {'learning_rate': 0.0005},
}
PROMPTS = ['what did the', 'why did the', 'two artists had',
           'my invisible penguin', 'zebra astronauts juggle']


def configurations(task, epochs=None):
    base = dict(embedding_dim=100, hidden_size=128, num_layers=1,
                dropout=0.2 if task == 'generation' else 0.3,
                max_epoch=epochs or (15 if task == 'generation' else 10),
                learning_rate=0.001, batch_size=64, grad_clip=5.0,
                optimizer_name='adam', seed=2)
    if task == 'generation':
        base['seq_len'] = 3
    else:
        base.update(bidirectional=False, track_test_acc=True)
    answer = {}
    for name, change in VARIANTS.items():
        cfg = {**base, **change}
        if name == 'sequence' and task == 'generation':
            cfg['seq_len'] = 5
        answer[name] = (cfg, 300 if name == 'sequence' and task == 'classification' else 200)
    return answer


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def digest_data(data):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        if split not in data:
            continue
        for key in ('X', 'y'):
            a = data[split][key]
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
    h.update(json.dumps(data['vocab']['id_to_word']).encode())
    return h.hexdigest()


def plot_history(record, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    h = record['history']
    x = range(1, len(h['loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(x, h['loss'], label='training')
    axes[0].set(xlabel='Epoch', ylabel='Cross entropy', title='Training loss')
    axes[1].plot(x, h['train_acc'], label='training, dropout on')
    if h.get('validation_acc'):
        axes[1].plot(x, h['validation_acc'], label='validation, dropout off')
    axes[1].set(xlabel='Epoch', ylabel='Accuracy', title='Next token' if record['task'] == 'generation' else 'Sentiment')
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(record['run_id'])
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def refresh_summary(out):
    records = [json.loads(p.read_text()) for p in sorted(out.glob('*/*/record.json'))]
    if not records:
        return
    fields = ['task','cell_type','variation','embedding_dim','hidden_size','num_layers',
              'seq_len','max_len','learning_rate','max_epoch','dropout','batch_size',
              'seed','embedding_source','recurrent_dropout','device','training_examples',
              'params','minutes','final_train_loss','final_train_acc',
              'train_eval_acc','validation_accuracy','accuracy','precision_macro','recall_macro','f1_macro']
    with (out / 'part_d_results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for r in records:
            writer.writerow({**r, **r['config'], **r['metrics']})
    write_json(out / 'part_d_history.json', {r['run_id']: r for r in records})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for task in sorted({r['task'] for r in records}):
        present_cells = [cell for cell in ('rnn', 'lstm', 'gru')
                         if any(r['task'] == task and r['cell_type'] == cell for r in records)]
        fig, axes = plt.subplots(1, len(present_cells), figsize=(4.5 * len(present_cells), 3.5), squeeze=False)
        for ax, cell in zip(axes[0], present_cells):
            for r in records:
                if r['task'] == task and r['cell_type'] == cell:
                    h = r['history']['loss']
                    ax.plot(range(1, len(h)+1), h, label=r['variation'])
            ax.set(title=cell.upper(), xlabel='Epoch', ylabel='Training cross entropy')
            ax.grid(alpha=0.25)
            if ax.lines:
                ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f'{task}_ablation_curves.png', dpi=160)
        plt.close(fig)


def run_task(task, args, out):
    import numpy as np
    import torch
    from sklearn.model_selection import train_test_split
    from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
    from local_code.stage_4_code.Dataset_Loader_Classification import Dataset_Loader_Classification
    from local_code.stage_4_code.Dataset_Loader_Generation import Dataset_Loader_Generation
    from local_code.stage_4_code.Method_RNN_Classification import Method_RNN_Classification
    from local_code.stage_4_code.Method_RNN_Generation import Method_RNN_Generation
    from local_code.stage_4_code.Glove_Utils import load_glove_matrix
    from run_generation_experiments import prepare_joke_index, compare_story

    cache, glove_cache = {}, {}
    for variation, (base_cfg, max_len) in configurations(task, args.epochs).items():
        if variation not in args.variants:
            continue
        cfg = {**base_cfg, 'device_name': args.device}
        if args.quick:
            cfg.update(max_epoch=1, train_subset=256)
        cache_key = cfg.get('seq_len', max_len)
        if cache_key not in cache:
            loader = Dataset_Loader_Generation() if task == 'generation' else Dataset_Loader_Classification()
            if task == 'generation':
                loader.context_len = cfg['seq_len']
            else:
                loader.max_len = max_len
            if args.data_dir:
                loader.dataset_source_folder_path = Path(args.data_dir) / ('text_generation/data' if task == 'generation' else 'text_classification')
            data = loader.load()
            data_hash = digest_data(data)
            original_test = None
            if task == 'classification':
                original_test = data['test']
                ids = np.arange(len(data['train']['y']))
                train_ids, val_ids = train_test_split(ids, test_size=0.1, random_state=2, stratify=data['train']['y'])
                original_train = data['train']
                data = {**data, 'train': {k: v[train_ids] for k,v in original_train.items()},
                        'test': {k: v[val_ids] for k,v in original_train.items()}}
                if args.quick:
                    # Smoke checks exercise scoring on small, stratified subsets.
                    for split in ('validation', 'test'):
                        source = data['test'] if split == 'validation' else original_test
                        if len(source['y']) > 256:
                            keep, _ = train_test_split(np.arange(len(source['y'])), train_size=256,
                                                       random_state=2, stratify=source['y'])
                            limited = {k: v[keep] for k, v in source.items()}
                            if split == 'validation':
                                data['test'] = limited
                            else:
                                original_test = limited
            cache[cache_key] = (loader, data, data_hash, original_test)
        loader, data, data_hash, original_test = cache[cache_key]
        embedding_source = 'learned from scratch'
        glove = None
        if task == 'classification' and args.glove:
            if cache_key not in glove_cache:
                glove_cache[cache_key] = load_glove_matrix(args.glove, data['vocab']['id_to_word'])
            glove = glove_cache[cache_key]
            if glove[0].shape[1] != cfg['embedding_dim']:
                raise ValueError('GloVe dimension must equal --embedding-dim baseline of 100. Use matching 100d vectors, or omit --glove.')
            embedding_source = f'GloVe {glove[0].shape[1]}d fine tuned'
        for cell in args.cells:
            config = {**cfg, 'cell_type': cell}
            run_id = f'{task}_{cell}_{variation}'
            folder = out / task / f'{cell}_{variation}'
            identity = {'config': config, 'max_len': max_len if task == 'classification' else None,
                        'data_sha256': data_hash, 'embedding_source': embedding_source,
                        'glove_sha256': args.glove_sha256 if task == 'classification' and args.glove else None,
                        'validation_fraction': 0.1 if task == 'classification' else None,
                        'recurrent_dropout': 0.0}
            if args.quick:
                identity['quick_eval_cap'] = 256
            if (folder / 'record.json').exists():
                saved = json.loads((folder / 'record.json').read_text())
                if saved['identity'] != identity:
                    raise ValueError(f'{run_id}: saved settings/data differ; choose a new --out directory.')
                print('Skipping completed', run_id, flush=True)
                continue
            folder.mkdir(parents=True, exist_ok=True)
            print('\nRUN', run_id, json.dumps(config), flush=True)
            method_class = Method_RNN_Generation if task == 'generation' else Method_RNN_Classification
            model = method_class(run_id, 'Part D single factor experiment', config=config)
            model.data = data
            if glove is not None:
                model.embedding_matrix = glove[0]
            # Keep recurrent dropout zero at BOTH depths. Embedding/output dropout stays fixed.
            model.build_model(data['vocab']['size'])
            model.rnn.dropout = 0.0
            if model.device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            result = model.run()
            metrics = {}
            if task == 'generation':
                metrics['train_eval_acc'] = model.final_eval_acc
                index = prepare_joke_index(data['jokes'])
                stories = []
                for prompt in PROMPTS:
                    for temperature in (None, 0.8):
                        generated = model.generate(prompt, temperature=temperature)
                        stories.append({'prompt': prompt, 'temperature': temperature,
                                        'text': loader.ids_to_text(generated['ids']),
                                        'unknown_words': generated['unknown_words'],
                                        'stopped_on': generated['stopped_on'],
                                        'comparison': compare_story(generated['words'], len(generated['prompt_words']), index)})
                write_json(folder / 'stories.json', stories)
            else:
                evaluator = Evaluate_Metrics('Part D evaluation', '')
                evaluator.data = result
                metrics['validation_accuracy'] = float(evaluator.evaluate()['accuracy'])
                # The original test set is scored once, after the fixed final epoch.
                # Never use it for per-epoch curves or choosing a checkpoint.
                pred = model.test(original_test['X'])
                evaluator.data = {'true_y': original_test['y'], 'pred_y': pred}
                metrics.update({k: float(v) for k,v in evaluator.evaluate().items()})
                np.savez_compressed(folder / 'predictions.npz', true_y=original_test['y'], pred_y=pred)
            if model.device.type == 'cuda':
                torch.cuda.synchronize()
            record = {'run_id': run_id, 'task': task, 'cell_type': cell, 'variation': variation,
                      'config': config, 'identity': identity, 'max_len': identity['max_len'],
                      'embedding_source': embedding_source, 'recurrent_dropout': 0.0,
                      'device': str(model.device), 'torch_version': torch.__version__,
                      'validation_examples': len(data['test']['y']) if task == 'classification' else 0,
                      'test_examples': len(original_test['y']) if task == 'classification' else 0,
                      'glove_stats': glove[1] if glove is not None else None,
                      'training_examples': min(len(data['train']['y']), config.get('train_subset') or len(data['train']['y'])),
                      'params': sum(p.numel() for p in model.parameters()),
                      'minutes': (time.perf_counter()-start)/60,
                      'final_train_loss': model.loss_history[-1], 'final_train_acc': model.train_acc_history[-1],
                      'metrics': metrics, 'history': {'loss': model.loss_history,
                      'train_acc': model.train_acc_history, 'validation_acc': getattr(model, 'test_acc_history', []),
                      'epoch_seconds': model.epoch_times}}
            if args.save_weights:
                torch.save({'state_dict': model.state_dict(), 'config': config, 'vocab': data['vocab'],
                            'identity': identity}, folder / 'weights.pt')
            plot_history(record, folder / 'learning_curve.png')
            write_json(folder / 'record.json', record)
            refresh_summary(out)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=['generation','classification','both'], default='generation')
    p.add_argument('--cells', nargs='+', choices=['rnn','lstm','gru'], default=['rnn','lstm','gru'])
    p.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=list(VARIANTS))
    p.add_argument('--epochs', type=int)
    p.add_argument('--data-dir', help='path containing text_classification/ and text_generation/')
    p.add_argument('--out', type=Path)
    p.add_argument('--device', default='auto', choices=['auto','cpu','cuda','mps'])
    p.add_argument('--threads', type=int, default=2, help='CPU threads; avoid excessive thread overhead on short sequences')
    p.add_argument('--glove', help='optional 100d GloVe for classification; omit embedding50 variant')
    p.add_argument('--save-weights', action='store_true')
    p.add_argument('--quick', action='store_true', help='one epoch on 256 training examples; smoke check only')
    p.add_argument('--plan', action='store_true', help='print experiment settings without loading data')
    args = p.parse_args()
    tasks = ['generation','classification'] if args.task == 'both' else [args.task]
    if args.epochs is not None and args.epochs < 1:
        p.error('--epochs must be positive')
    if args.threads < 1:
        p.error('--threads must be positive')
    if args.glove and 'classification' in tasks and 'embedding50' in args.variants:
        p.error('With GloVe select --variants baseline hidden256 layers2 sequence lr0005. Compare embedding sizes separately with matched initialization.')
    if args.plan:
        print(json.dumps({t: {k:{'model': c,'max_len': m if t=='classification' else None}
              for k,(c,m) in configurations(t,args.epochs).items() if k in args.variants} for t in tasks}, indent=2))
        return
    import torch
    torch.set_num_threads(args.threads)
    args.glove_sha256 = sha256_file(args.glove) if args.glove else None
    out = args.out or ROOT / 'result/stage_4_result' / ('part_d_quick' if args.quick else 'part_d')
    out.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        run_task(task, args, out)
    refresh_summary(out)
    print('Saved:', out.resolve())


if __name__ == '__main__':
    main()

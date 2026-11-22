'''
Part C: Stage 4 joke generation with RNN, LSTM and GRU.

Trains each cell type on Part A's next-word windows, saves a learning curve for
each, one overlaid comparison plot, the training history (for Part E), and
generated stories with a training-set overlap check.

Run from the repo root (or press Run in PyCharm), after data/stage_4_data/ is unzipped:
  python script/stage_4_script/run_generation_experiments.py                 # rnn, lstm, gru
  python script/stage_4_script/run_generation_experiments.py --only lstm
  python script/stage_4_script/run_generation_experiments.py --quick         # 1 epoch, 2,000 pairs
  python script/stage_4_script/run_generation_experiments.py --epochs 30

CPU defaults are embedding 100, hidden 128, 1 layer, seq_len 3, 15 epochs, batch 64.
On Colab, raise capacity once a GPU is in use, for example:
  python script/stage_4_script/run_generation_experiments.py --epochs 30 --hidden-size 256 --layers 2 --embedding-dim 128 --batch-size 256

Part B's plotting functions live inside its script (there is no shared helper),
so the curves here follow that same two-panel layout.
'''

import argparse
import csv
import json
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

from local_code.stage_4_code.Dataset_Loader_Generation import Dataset_Loader_Generation
from local_code.stage_4_code.Method_RNN_Generation import Method_RNN_Generation

RESULT_DIR = ROOT / 'result' / 'stage_4_result' / 'generation'

# Same keys as Part B where the two models overlap. seq_len is the generation window.
SHARED = {
    'embedding_dim': 100,
    'hidden_size': 128,
    'num_layers': 1,
    'seq_len': 3,
    'dropout': 0.2,
    'max_epoch': 15,
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

# Fixed prompts. The first three are real joke openings (many training jokes start
# this way, except "two artists had", which is one specific joke). The last two are
# not in the training set; some of their words are out of vocabulary and become <unk>.
TRAINING_PROMPTS = [
    ['what', 'did', 'the'],
    ['why', 'did', 'the'],
    ['two', 'artists', 'had'],
]
NEW_PROMPTS = [
    ['my', 'invisible', 'penguin'],   # words are in the vocab, this opening is not
    ['zebra', 'astronauts', 'juggle'],  # none of these words are in the vocab, so they become <unk>
]
# A copied span at least this many tokens long counts as a long verbatim chunk.
LONG_CHUNK = 6
TOKEN_SEP = '\u241f'
TEMPERATURE = 0.8
MAX_GEN_LEN = 40


def save_learning_curves(m, out_path, title):
    '''Two panels, same layout as Part B's classification curves.'''
    epochs = np.arange(1, len(m.loss_history) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, m.loss_history, marker='o', label='train loss')
    axes[0].set(title='Training Loss', xlabel='Epoch', ylabel='Cross-entropy loss')
    axes[0].legend()
    axes[1].plot(epochs, m.train_acc_history, marker='o', label='train acc')
    axes[1].set(title='Next-word Accuracy', xlabel='Epoch', ylabel='Accuracy')
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.set_xticks(epochs)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_comparison_plot(history, out_path):
    '''One figure with every trained cell's loss and accuracy on shared axes.'''
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, record in history.items():
        epochs = np.arange(1, len(record['loss']) + 1)
        label = name.upper()
        axes[0].plot(epochs, record['loss'], marker='o', label=label)
        axes[1].plot(epochs, record['train_acc'], marker='o', label=label)
    axes[0].set(title='Training Loss', xlabel='Epoch', ylabel='Cross-entropy loss')
    axes[1].set(title='Next-word Accuracy', xlabel='Epoch', ylabel='Accuracy')
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle('Joke generation: RNN vs LSTM vs GRU')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def append_row(csv_path, row):
    rows = []
    if csv_path.exists():
        with csv_path.open() as f:
            rows = [r for r in csv.DictReader(f) if r['name'] != row['name']]
    rows.append({k: str(v) for k, v in row.items()})
    order = {'rnn': 0, 'lstm': 1, 'gru': 2}
    rows.sort(key=lambda r: order.get(r['name'], 9))
    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerows(rows)


def _marked(tokens):
    '''Join tokens so a substring match is an exact token sequence, not a prefix of a word.'''
    return TOKEN_SEP + TOKEN_SEP.join(tokens) + TOKEN_SEP


def longest_verbatim(gen_tokens, joke_strings):
    '''Longest contiguous piece of gen_tokens that appears inside some training joke.

    Returns (length, start_index_in_gen, joke_index) or (0, 0, -1).
    '''
    if not gen_tokens:
        return 0, 0, -1
    for n in range(len(gen_tokens), 0, -1):
        for start in range(len(gen_tokens) - n + 1):
            chunk = _marked(gen_tokens[start:start + n])
            for ji, joke_s in enumerate(joke_strings):
                if chunk in joke_s:
                    return n, start, ji
    return 0, 0, -1


def ngram_set(tokens, n):
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def closest_joke(gen_tokens, joke_ngrams, n):
    '''Training joke that shares the largest fraction of this story's n-grams.

    A score of 0 is reported as "no joke": sharing nothing is not a match.
    '''
    generated = ngram_set(gen_tokens, n)
    if not generated:
        return -1, 0.0
    best_i, best_score = -1, 0.0
    for i, joke_grams in enumerate(joke_ngrams):
        if not joke_grams:
            continue
        score = len(generated & joke_grams) / len(generated)
        if score > best_score:
            best_i, best_score = i, score
    return best_i, best_score


def prepare_joke_index(jokes):
    return {
        'strings': [_marked(j) for j in jokes],
        'tri': [ngram_set(j, 3) for j in jokes],
    }


def compare_story(gen_tokens, prompt_len, index):
    '''Verbatim check plus the closest training joke by n-gram overlap.

    The prompt itself is often copied from the training set, so the continuation
    (everything after those three words) is scored on its own as well.
    '''
    strings = index['strings']
    full = _marked(gen_tokens) if gen_tokens else ''
    continuation = gen_tokens[prompt_len:]
    cont_marked = _marked(continuation) if continuation else ''
    full_hits = [i for i, s in enumerate(strings) if full and full in s]
    cont_hits = [i for i, s in enumerate(strings) if len(continuation) >= 4 and cont_marked in s]
    length, start, joke_i = longest_verbatim(gen_tokens, strings)
    # Longest copied span of the continuation, so the 3-word prompt alone
    # does not count as the model memorizing a joke.
    cont_len, cont_start, cont_joke = longest_verbatim(continuation, strings) if continuation else (0, 0, -1)
    # Trigrams only. Single shared words ("the", "a") match almost every joke.
    close_i, close_score = closest_joke(gen_tokens, index['tri'], 3) if len(gen_tokens) >= 3 else (-1, 0.0)
    if len(continuation) >= 3:
        close_c_i, close_c_score = closest_joke(continuation, index['tri'], 3)
    else:
        close_c_i, close_c_score = -1, 0.0
    return {
        'verbatim_full_story': bool(full_hits),
        'verbatim_full_joke_indexes': full_hits[:5],
        'verbatim_continuation': bool(cont_hits),
        'verbatim_continuation_joke_indexes': cont_hits[:5],
        'longest_chunk_tokens': length,
        'longest_chunk_text': ' '.join(gen_tokens[start:start + length]) if length else '',
        'longest_chunk_joke': joke_i,
        'longest_continuation_chunk_tokens': cont_len,
        'longest_continuation_chunk_text': ' '.join(continuation[cont_start:cont_start + cont_len]) if cont_len else '',
        'longest_continuation_chunk_joke': cont_joke,
        'long_chunk_copied': cont_len >= LONG_CHUNK,
        'closest_joke': close_i,
        'closest_overlap': round(close_score, 4),
        'closest_continuation_joke': close_c_i,
        'closest_continuation_overlap': round(close_c_score, 4),
    }


def joke_preview(loader, joke_index):
    if joke_index is None or joke_index < 0:
        return 'none'
    return f'joke {joke_index}: {loader.ids_to_text(loader.joke_ids[joke_index])}'


def format_story_block(name, decoding, story, comparison, loader, prompt_hits):
    prompt = ' '.join(story['prompt_words'])
    text = loader.ids_to_text(story['ids'])
    unk = ', '.join(story['unknown_words']) if story['unknown_words'] else 'none'
    lines = [
        f'===== {name.upper()} ({decoding}) =====',
        f'prompt: {prompt}',
        f'prompt is the opening of {prompt_hits} training joke(s)',
        f'unknown prompt words: {unk}',
        f'story: {text}',
        f'stopped: {story["stopped_on"]} ({len(story["generated_ids"])} new words)',
        f'verbatim full story in training: {"yes" if comparison["verbatim_full_story"] else "no"}'
        + (f' (joke {comparison["verbatim_full_joke_indexes"][0]})' if comparison['verbatim_full_story'] else ''),
        f'verbatim continuation in training: {"yes" if comparison["verbatim_continuation"] else "no"}'
        + (f' (joke {comparison["verbatim_continuation_joke_indexes"][0]})' if comparison['verbatim_continuation'] else ''),
        f'longest copied chunk: {comparison["longest_chunk_tokens"]} words'
        + (f' "{comparison["longest_chunk_text"]}" (joke {comparison["longest_chunk_joke"]})'
           if comparison['longest_chunk_tokens'] else ''),
        f'longest copied chunk after the prompt: {comparison["longest_continuation_chunk_tokens"]} words'
        + (f' "{comparison["longest_continuation_chunk_text"]}" (joke {comparison["longest_continuation_chunk_joke"]})'
           if comparison['longest_continuation_chunk_tokens'] else ''),
        f'long chunk copied ({LONG_CHUNK}+ words after the prompt): '
        f'{"yes" if comparison["long_chunk_copied"] else "no"}',
        'closest training joke by story trigram overlap: '
        + (joke_preview(loader, comparison['closest_joke'])
           + f' (overlap {comparison["closest_overlap"]:.3f})'
           if comparison['closest_joke'] >= 0 else 'none'),
        'closest training joke by continuation trigram overlap: '
        + (joke_preview(loader, comparison['closest_continuation_joke'])
           + f' (overlap {comparison["closest_continuation_overlap"]:.3f})'
           if comparison['closest_continuation_joke'] >= 0 else 'none (continuation has no shared trigram)'),
        '',
    ]
    return '\n'.join(lines)


def load_json(path, default):
    if not path.exists():
        return default
    with path.open() as f:
        return json.load(f)


def save_json(path, obj):
    with path.open('w') as f:
        json.dump(obj, f, indent=2)
        f.write('\n')


def run_one(exp, data, loader, jokes, joke_index, result_dir, overrides, max_len, temperature):
    name = exp['name']
    config = {**SHARED, **exp['config'], **overrides}
    print('\n' + '=' * 60)
    print(f"Run: {name}  (cell_type={config['cell_type']})")
    print('=' * 60, flush=True)

    method_obj = Method_RNN_Generation(f'Jokes {name.upper()}', '', config=config)
    method_obj.data = data

    start = time.time()
    method_obj.run()
    minutes = (time.time() - start) / 60
    method_obj.print_parameter_table()

    curve = result_dir / f'GEN_learning_curve_{name}.png'
    save_learning_curves(method_obj, curve, f'Joke generation, {name.upper()}')
    print('Saved learning curve:', curve)

    prompts = [('training', p) for p in TRAINING_PROMPTS] + [('new', p) for p in NEW_PROMPTS]
    stories = []
    for prompt_index, (source, prompt) in enumerate(prompts):
        prompt_hits = sum(1 for joke in jokes if joke[:len(prompt)] == prompt)
        for decoding, temp in (('greedy', None), (f'temperature {temperature}', temperature)):
            generated = method_obj.generate(prompt, max_len=max_len, temperature=temp)
            comparison = compare_story(generated['words'], len(generated['prompt_words']), joke_index)
            block = format_story_block(name, decoding, generated, comparison, loader, prompt_hits)
            stories.append({
                'model': name,
                'source': source,
                'prompt_index': prompt_index,
                'decoding': decoding,
                'prompt': generated['prompt_words'],
                'prompt_hits': prompt_hits,
                'text': loader.ids_to_text(generated['ids']),
                'unknown_words': generated['unknown_words'],
                'stopped_on': generated['stopped_on'],
                'new_words': len(generated['generated_ids']),
                'comparison': comparison,
                'report': block,
            })
            print(block)

    row = {
        'name': name,
        'cell_type': config['cell_type'],
        'embedding_dim': config['embedding_dim'],
        'hidden_size': config['hidden_size'],
        'num_layers': config['num_layers'],
        'seq_len': config['seq_len'],
        'dropout': config['dropout'],
        'max_epoch': config['max_epoch'],
        'learning_rate': config['learning_rate'],
        'batch_size': config['batch_size'],
        'grad_clip': config['grad_clip'],
        'vocab_size': method_obj.vocab_size,
        'training_pairs': len(data['train']['y']) if not config.get('train_subset') else config['train_subset'],
        'params': sum(r[2] for r in method_obj.parameter_table()),
        'device': str(method_obj.device),
        'minutes': round(minutes, 2),
        'final_train_loss': method_obj.loss_history[-1],
        'final_train_acc': method_obj.train_acc_history[-1],
        'final_eval_acc': method_obj.final_eval_acc,
        'learning_curve': curve.name,
    }
    history_record = {
        'cell_type': config['cell_type'],
        'loss': [float(v) for v in method_obj.loss_history],
        'train_acc': [float(v) for v in method_obj.train_acc_history],
        'epoch_seconds': [float(v) for v in method_obj.epoch_times],
        'final_eval_acc': method_obj.final_eval_acc,
        'config': {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in config.items()},
    }
    return row, history_record, stories


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--only', nargs='+', choices=[e['name'] for e in EXPERIMENTS],
                        help='train these cells only (default: rnn lstm gru)')
    parser.add_argument('--epochs', type=int, help='override max_epoch for every run')
    parser.add_argument('--embedding-dim', type=int)
    parser.add_argument('--hidden-size', type=int)
    parser.add_argument('--layers', type=int, help='override num_layers')
    parser.add_argument('--seq-len', type=int, help='context window; also sets the loader context_len')
    parser.add_argument('--lr', type=float, help='override learning_rate')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--max-len', type=int, default=MAX_GEN_LEN, help='most new words per story')
    parser.add_argument('--temperature', type=float, default=TEMPERATURE)
    parser.add_argument('--quick', action='store_true',
                        help='1 epoch on 2,000 training pairs, results go to generation_quick/')
    parser.add_argument('--cpu', action='store_true', help='force CPU even if a GPU exists')
    args = parser.parse_args()

    from run_log import start_log
    start_log('run_generation_experiments' + ('_quick' if args.quick else ''))

    overrides, result_dir = {}, RESULT_DIR
    if args.quick:
        overrides.update({'max_epoch': 1, 'train_subset': 2000})
        result_dir = RESULT_DIR.parent / 'generation_quick'
    if args.epochs:
        overrides['max_epoch'] = args.epochs
    if args.embedding_dim:
        overrides['embedding_dim'] = args.embedding_dim
    if args.hidden_size:
        overrides['hidden_size'] = args.hidden_size
    if args.layers:
        overrides['num_layers'] = args.layers
    if args.seq_len:
        overrides['seq_len'] = args.seq_len
    if args.lr:
        overrides['learning_rate'] = args.lr
    if args.batch_size:
        overrides['batch_size'] = args.batch_size
    if args.cpu:
        overrides['device_name'] = 'cpu'
    result_dir.mkdir(parents=True, exist_ok=True)

    seq_len = overrides.get('seq_len', SHARED['seq_len'])
    print('************ Start joke generation experiments ************')
    loader = Dataset_Loader_Generation()
    loader.context_len = seq_len
    data = loader.load()
    jokes = data['jokes']
    print(f"jokes {len(jokes)}, vocab {data['vocab']['size']}, pairs {data['train']['X'].shape}, "
          f"context_len {data['context_len']}", flush=True)
    joke_index = prepare_joke_index(jokes)

    experiments = [e for e in EXPERIMENTS if not args.only or e['name'] in args.only]
    csv_path = result_dir / 'GEN_experiment_results.csv'
    history_path = result_dir / 'GEN_training_history.json'
    stories_path = result_dir / 'GEN_stories.txt'
    stories_json_path = result_dir / 'GEN_stories.json'
    history = load_json(history_path, {})
    previous_stories = load_json(stories_json_path, [])

    for exp in experiments:
        row, record, stories = run_one(
            exp, data, loader, jokes, joke_index, result_dir, overrides, args.max_len, args.temperature)
        append_row(csv_path, row)
        history[exp['name']] = record
        save_json(history_path, history)
        kept = [s for s in previous_stories if s.get('model') != exp['name']]
        previous_stories = kept + stories
        save_json(stories_json_path, previous_stories)
        print('Updated', csv_path, flush=True)

    # Rewrite the text report from the merged JSON so a partial --only run keeps the other models.
    header = (
        'Joke generation stories\n'
        f'seq_len {seq_len}, max new words {args.max_len}, temperature {args.temperature}\n'
        'Greedy is the main comparison. Temperature samples are included under each model.\n'
        f'A "long chunk" is {LONG_CHUNK}+ generated words (after the 3-word prompt) found verbatim in a training joke.\n'
        'Joke indexes are 0-based positions in the loader, which reads the file in order '
        '(joke 0 is the first joke in the CSV).'
    )
    order = {'rnn': 0, 'lstm': 1, 'gru': 2}
    previous_stories.sort(key=lambda s: (order.get(s['model'], 9), s.get('prompt_index', 0), s['decoding']))
    save_json(stories_json_path, previous_stories)
    reports = [s.get('report') or s.get('text', '') for s in previous_stories]
    stories_path.write_text(header + '\n\n' + '\n'.join(reports), encoding='utf-8')
    print('Saved stories:', stories_path)

    save_comparison_plot(history, result_dir / 'GEN_loss_comparison.png')
    print('\n************ Results table ************')
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    print(f"{'cell':<6} {'loss':>8} {'acc':>8} {'eval':>8} {'min':>6}")
    for r in rows:
        print(f"{r['name']:<6} {float(r['final_train_loss']):8.4f} {float(r['final_train_acc']):8.4f} "
              f"{float(r['final_eval_acc']):8.4f} {float(r['minutes']):6.1f}")
    print('************ Finish ************')


if __name__ == '__main__':
    main()

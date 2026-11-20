'''
Person A: check both Stage 4 loaders and print the numbers to post in the group chat.

Run from the repo root (or press Run in PyCharm):
  python3 script/stage_4_script/check_text_data.py
'''

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_code.stage_4_code.Dataset_Loader_Classification import Dataset_Loader_Classification
from local_code.stage_4_code.Dataset_Loader_Generation import Dataset_Loader_Generation, EOS_ID


def check(condition, message):
    if not condition:
        raise SystemExit(f'FAIL: {message}')


def check_classification():
    print('=' * 60)
    print('Text classification (IMDB reviews)')
    print('=' * 60)
    loader = Dataset_Loader_Classification()
    data = loader.load()
    s = loader.stats
    for split in ('train', 'test'):
        X, y, lengths = data[split]['X'], data[split]['y'], data[split]['lengths']
        counts = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
        print(f'{split}: X={X.shape} ({X.dtype}), y={y.shape} ({y.dtype}), label counts {counts} (0=neg, 1=pos)')
        check(X.shape == (25000, loader.max_len), f'{split} X shape')
        check(counts == {0: 12500, 1: 12500}, f'{split} label counts')
        check(X.min() >= 0 and X.max() < s['vocab_size'], f'{split} word IDs outside the vocabulary')
        check((lengths > 0).all(), f'{split} has an empty review')
        # pre-padding: zeros only at the start of each row
        first_word = loader.max_len - lengths
        check(all((X[i, :first_word[i]] == 0).all() and (X[i, first_word[i]:] != 0).all()
                  for i in range(0, len(X), 997)), f'{split} padding is not at the start')
        neg_ok = (data[split]['ratings'][y == 0] <= 4).all()
        pos_ok = (data[split]['ratings'][y == 1] >= 7).all()
        check(neg_ok and pos_ok, f'{split} labels do not match the star ratings in the file names')

    print(f"vocab size: {s['vocab_size']} (0=<pad>, 1=<unk>; {s['distinct_train_words']} distinct words in train,"
          f' kept those seen >= {loader.min_freq} times, capped at {loader.max_vocab_size})')
    print(f"max length: {s['max_len']} words (padding at the start, long reviews keep their last {loader.max_len} words)")
    print(f"train review length: mean {s['train_review_words_mean']:.0f}, median {s['train_review_words_median']:.0f},"
          f" max {s['train_review_words_max']}; {s['train_reviews_cut']} of 25000 were cut")
    print(f"test words not in vocab: {s['test_unknown_word_rate']:.1%}")
    row = data['train']
    print(f"example (label {row['y'][0]}): {loader.decode(row['X'][0])[:200]}...")
    print('PASS: classification loader')


def check_generation():
    print('\n' + '=' * 60)
    print('Text generation (jokes)')
    print('=' * 60)
    loader = Dataset_Loader_Generation()
    data = loader.load()
    s = loader.stats
    X, y = data['train']['X'], data['train']['y']
    print(f"jokes: {s['jokes']}, vocab size: {s['vocab_size']} (0=<pad>, 1=<unk>, 2=<eos>),"
          f" {s['words_seen_once']} words appear only once")
    print(f"joke length: mean {s['joke_words_mean']:.1f}, max {s['joke_words_max']} tokens")
    print(f"training pairs: X={X.shape} ({X.dtype}), y={y.shape} ({y.dtype}), context_len={loader.context_len}")
    check(s['jokes'] == 1622, 'expected 1622 jokes')
    check(len(y) == sum(len(j) for j in data['joke_ids']), 'one training pair per word + <eos>')
    check((y == EOS_ID).sum() == s['jokes'], 'one <eos> per joke')
    check(X.max() < s['vocab_size'] and y.max() < s['vocab_size'], 'IDs outside the vocabulary')

    original = ' '.join(data['jokes'][0])
    round_trip = ' '.join(loader.ids_to_words(data['joke_ids'][0]))
    check(original == round_trip, 'words -> IDs -> words round trip changed the joke')
    print(f"joke 1 tokens: {data['jokes'][0]}")
    print(f"joke 1 as text: {loader.ids_to_text(data['joke_ids'][0])}")
    print('first training pairs of joke 1:')
    for i in range(4):
        print(f"  {loader.ids_to_words(X[i], stop_at_eos=False) or ['<pad>']} -> "
              f"{loader.id_to_word[y[i]]}")
    start = loader.words_to_ids('What did the')
    print(f"words_to_ids('What did the') = {start} -> {loader.ids_to_words(start)}")
    print(f"unknown_words(['what', 'did', 'zebrafish']) = {loader.unknown_words(['what', 'did', 'zebrafish'])}")
    print('PASS: generation loader')


if __name__ == '__main__':
    check_classification()
    check_generation()
    print('\nPASS: both Stage 4 loaders')

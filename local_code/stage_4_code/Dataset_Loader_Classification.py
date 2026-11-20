'''
Stage 4 loader for the IMDB movie-review sentiment dataset (text classification).

Reads data/stage_4_data/text_classification/{train,test}/{pos,neg}/*.txt and returns

    {
        'train': {'X': int64 (25000, max_len), 'y': int64 (25000,), 'lengths': int64 (25000,)},
        'test':  {'X': int64 (25000, max_len), 'y': int64 (25000,), 'lengths': int64 (25000,)},
        'vocab': {'word_to_id': dict, 'id_to_word': list, 'size': int},
        'max_len': int,
    }

X holds word IDs (0 = <pad>, 1 = <unk>), y is 0 = negative, 1 = positive.
lengths is the number of real (non-pad) words in each row, capped at max_len.
'''

from pathlib import Path
import random

import numpy as np

from local_code.base_class.dataset import dataset
from local_code.stage_4_code.Text_Utils import (
    PAD, UNK, PAD_ID, build_vocab, encode, tokenize,
)

DATA_DIRECTORY = Path(__file__).resolve().parents[2] / 'data' / 'stage_4_data' / 'text_classification'


class Dataset_Loader_Classification(dataset):
    # ---- settings (change before calling load()) ----
    max_len = 200                # every review is padded / cut to this many words
    max_vocab_size = 20000       # including <pad> and <unk>
    min_freq = 2                 # words seen fewer times in TRAIN become <unk>
    remove_stop_words = False    # README suggests it; off by default (see Text_Utils.STOP_WORDS)
    # 'pre' = zeros go BEFORE the words, so the review's last word sits next to the RNN's final
    # hidden state. Use 'post' only if your model handles padding itself (e.g. packed sequences).
    padding = 'pre'
    # 'pre' keeps the LAST max_len words (endings often hold the verdict); 'post' keeps the first.
    truncating = 'pre'
    shuffle_seed = 2             # train rows are shuffled once so pos/neg are mixed; None = keep file order

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName or 'IMDB', dDescription or 'movie review sentiment')
        self.dataset_source_folder_path = DATA_DIRECTORY
        self.data = None
        self.stats = {}

    def _read_split(self, split):
        texts, labels, ratings = [], [], []
        root = Path(self.dataset_source_folder_path) / split
        for label_name, label in (('neg', 0), ('pos', 1)):
            folder = root / label_name
            if not folder.is_dir():
                raise FileNotFoundError(f'Missing folder {folder}. Put the Stage 4 data in data/stage_4_data/.')
            for path in sorted(folder.glob('*.txt')):
                texts.append(path.read_text(encoding='utf-8', errors='replace'))
                labels.append(label)
                ratings.append(int(path.stem.split('_')[1]))  # file name is <id>_<rating>.txt
        return texts, labels, ratings

    def _to_matrix(self, token_lists):
        X = np.full((len(token_lists), self.max_len), PAD_ID, dtype=np.int64)
        lengths = np.zeros(len(token_lists), dtype=np.int64)
        for row, tokens in enumerate(token_lists):
            ids = encode(tokens, self.word_to_id)
            if len(ids) > self.max_len:
                ids = ids[-self.max_len:] if self.truncating == 'pre' else ids[:self.max_len]
            if ids:
                if self.padding == 'pre':
                    X[row, -len(ids):] = ids
                else:
                    X[row, :len(ids)] = ids
            lengths[row] = len(ids)
        return X, lengths

    def load(self):
        print('loading data...')
        train_texts, train_y, train_ratings = self._read_split('train')
        test_texts, test_y, test_ratings = self._read_split('test')

        tok = dict(keep_punctuation=False, remove_stop_words=self.remove_stop_words)
        train_tokens = [tokenize(t, **tok) for t in train_texts]
        test_tokens = [tokenize(t, **tok) for t in test_texts]

        # vocabulary from TRAINING reviews only, so nothing leaks from the test set
        self.word_to_id, self.id_to_word, counts = build_vocab(
            train_tokens, special_tokens=(PAD, UNK),
            max_size=self.max_vocab_size, min_freq=self.min_freq)

        if self.shuffle_seed is not None:
            order = list(range(len(train_tokens)))
            random.Random(self.shuffle_seed).shuffle(order)
            train_tokens = [train_tokens[i] for i in order]
            train_y = [train_y[i] for i in order]
            train_ratings = [train_ratings[i] for i in order]

        X_train, len_train = self._to_matrix(train_tokens)
        X_test, len_test = self._to_matrix(test_tokens)

        raw_train_len = np.array([len(t) for t in train_tokens])
        test_word_total = sum(len(t) for t in test_tokens)
        test_unk = sum(1 for t in test_tokens for w in t if w not in self.word_to_id)
        self.stats = {
            'vocab_size': len(self.id_to_word),
            'distinct_train_words': len(counts),
            'max_len': self.max_len,
            'train_review_words_mean': float(raw_train_len.mean()),
            'train_review_words_median': float(np.median(raw_train_len)),
            'train_review_words_max': int(raw_train_len.max()),
            'train_reviews_cut': int((raw_train_len > self.max_len).sum()),
            'test_unknown_word_rate': test_unk / max(test_word_total, 1),
        }

        self.data = {
            'train': {'X': X_train, 'y': np.asarray(train_y, dtype=np.int64), 'lengths': len_train,
                      'ratings': np.asarray(train_ratings, dtype=np.int64)},
            'test': {'X': X_test, 'y': np.asarray(test_y, dtype=np.int64), 'lengths': len_test,
                     'ratings': np.asarray(test_ratings, dtype=np.int64)},
            'vocab': {'word_to_id': self.word_to_id, 'id_to_word': self.id_to_word,
                      'size': len(self.id_to_word)},
            'max_len': self.max_len,
        }
        return self.data

    def decode(self, ids):
        '''Word IDs -> readable text (padding dropped). Handy for checking a row.'''
        return ' '.join(self.id_to_word[i] for i in ids if i != PAD_ID)

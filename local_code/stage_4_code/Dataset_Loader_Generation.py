'''
Stage 4 loader for the short-jokes dataset (text generation).

Reads data/stage_4_data/text_generation/data (a CSV with columns "ID","Joke") and returns

    {
        'jokes':      list of word lists, one per joke (punctuation kept as its own token)
        'joke_ids':   list of ID lists, one per joke, each ending with <eos>
        'train':      {'X': int64 (N, context_len), 'y': int64 (N,)}   # next-word training pairs
        'vocab':      {'word_to_id': dict, 'id_to_word': list, 'size': int},
        'context_len': int,
    }

Vocabulary: 0 = <pad>, 1 = <unk>, 2 = <eos> (end of joke), 3+ = words.
Every word in the jokes is kept (no minimum frequency), so <unk> only appears for words the
model has never seen, e.g. starting words you type in yourself.

Training pairs: every position in every joke becomes one example. X is the previous
context_len words (left-padded with <pad> at the start of a joke) and y is the next word.
For "what did the bartender say ..." with context_len 3:
    [<pad> <pad> what] -> did,  [<pad> what did] -> the,  [what did the] -> bartender, ...
The last pair of each joke predicts <eos>, which tells the generator when to stop.
'''

from pathlib import Path
import csv

import numpy as np

from local_code.base_class.dataset import dataset
from local_code.stage_4_code.Text_Utils import (
    EOS, PAD, UNK, PAD_ID, UNK_ID, build_vocab, decode, encode, tokenize,
)

DATA_FILE = Path(__file__).resolve().parents[2] / 'data' / 'stage_4_data' / 'text_generation' / 'data'
EOS_ID = 2


class Dataset_Loader_Generation(dataset):
    # ---- settings (change before calling load()) ----
    context_len = 3        # words the model sees before predicting the next one
    keep_punctuation = True

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName or 'Jokes', dDescription or 'short jokes for text generation')
        self.dataset_source_folder_path = DATA_FILE
        self.data = None
        self.stats = {}

    def _read_jokes(self):
        path = Path(self.dataset_source_folder_path)
        if not path.is_file():
            raise FileNotFoundError(f'Missing {path}. Put the Stage 4 data in data/stage_4_data/.')
        with path.open(encoding='utf-8', newline='') as f:
            return [row['Joke'] for row in csv.DictReader(f) if row['Joke'].strip()]

    def load(self):
        print('loading data...')
        raw = self._read_jokes()
        self.jokes = [tokenize(j, keep_punctuation=self.keep_punctuation) for j in raw]
        self.word_to_id, self.id_to_word, counts = build_vocab(
            self.jokes, special_tokens=(PAD, UNK, EOS))
        self.joke_ids = [encode(words, self.word_to_id) + [EOS_ID] for words in self.jokes]
        X, y = self.make_training_pairs(self.context_len)

        lengths = np.array([len(j) for j in self.jokes])
        self.stats = {
            'jokes': len(self.jokes),
            'vocab_size': len(self.id_to_word),
            'words_seen_once': sum(1 for c in counts.values() if c == 1),
            'joke_words_mean': float(lengths.mean()),
            'joke_words_max': int(lengths.max()),
            'training_pairs': len(y),
        }
        self.data = {
            'jokes': self.jokes,
            'joke_ids': self.joke_ids,
            'train': {'X': X, 'y': y},
            'vocab': {'word_to_id': self.word_to_id, 'id_to_word': self.id_to_word,
                      'size': len(self.id_to_word)},
            'context_len': self.context_len,
        }
        return self.data

    def make_training_pairs(self, context_len):
        '''(X, y) next-word pairs for any context length, e.g. to try 5 instead of 3.'''
        X, y = [], []
        for ids in self.joke_ids:
            padded = [PAD_ID] * context_len + ids
            for t in range(len(ids)):
                X.append(padded[t:t + context_len])
                y.append(ids[t])
        return np.asarray(X, dtype=np.int64), np.asarray(y, dtype=np.int64)

    # ---- helpers for the generation model ----
    def words_to_ids(self, text_or_words):
        '''"What did the" or ['what', 'did', 'the'] -> [ids]. Cleans text the same way as training.'''
        words = (tokenize(text_or_words, keep_punctuation=self.keep_punctuation)
                 if isinstance(text_or_words, str) else [w.lower() for w in text_or_words])
        return encode(words, self.word_to_id)

    def ids_to_words(self, ids, stop_at_eos=True):
        '''[ids] -> list of words. Stops at the first <eos> unless stop_at_eos=False.'''
        ids = list(ids)
        if stop_at_eos and EOS_ID in ids:
            ids = ids[:ids.index(EOS_ID)]
        return decode(ids, self.id_to_word)

    def ids_to_text(self, ids, stop_at_eos=True):
        '''[ids] -> readable string with punctuation attached to the word before it.'''
        text = ' '.join(self.ids_to_words(ids, stop_at_eos))
        for p in '.!?,;:':
            text = text.replace(' ' + p, p)
        return text

    def unknown_words(self, words):
        '''Which of these starting words are not in the vocabulary (they would become <unk>).'''
        return [w for w in words if w.lower() not in self.word_to_id]

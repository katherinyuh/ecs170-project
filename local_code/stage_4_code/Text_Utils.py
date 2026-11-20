'''
Shared text cleaning and vocabulary helpers for the Stage 4 loaders.

Vocabulary convention used by every Stage 4 loader:
    0 = <pad>   (padding; models should treat it as "no word")
    1 = <unk>   (any word not in the vocabulary)
    2+          real words, most frequent first
The generation vocabulary also adds 2 = <eos> (end of joke), so its real words start at 3.
'''

from collections import Counter
import html
import re

PAD, UNK, EOS = '<pad>', '<unk>', '<eos>'
PAD_ID, UNK_ID = 0, 1

_HTML_TAG = re.compile(r'<[^>]+>')  # IMDB reviews contain <br /> line breaks
_NON_WORD = re.compile(r"[^a-z0-9' ]+")
_WORD_OR_PUNCT = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*|[.!?,;:]")

# Common English stop words. Negations and contrast words ("not", "no", "but", ...) are
# deliberately left OUT of this list: "not good" and "good" mean opposite things for sentiment.
STOP_WORDS = frozenset('''
a an the and or of to in on at by for with from as is am are was were be been being
this that these those it its it's i me my we our you your he him his she her they them their
there here what which who whom will would shall should can could may might must do does did
done have has had having so than too very just also into onto about over under again then
once only own same such s t ll d m re ve
'''.split())


def clean_text(text, keep_punctuation=False):
    '''Lowercase, unescape HTML entities, drop HTML tags, and normalize spacing.

    keep_punctuation=False: strip everything except letters, digits and apostrophes
        (used for classification).
    keep_punctuation=True: keep . ! ? , ; : as separate tokens
        (used for generation, so generated jokes keep their question marks and punchlines).
    '''
    text = html.unescape(text)
    text = _HTML_TAG.sub(' ', text).lower()
    text = text.replace('’', "'").replace('‘', "'")
    if keep_punctuation:
        return ' '.join(_WORD_OR_PUNCT.findall(text))
    text = _NON_WORD.sub(' ', text)
    # trim apostrophes left at word edges ('quoted' -> quoted) but keep don't, it's
    return ' '.join(w.strip("'") for w in text.split() if w.strip("'"))


def tokenize(text, keep_punctuation=False, remove_stop_words=False):
    words = clean_text(text, keep_punctuation=keep_punctuation).split()
    if remove_stop_words:
        words = [w for w in words if w not in STOP_WORDS]
    return words


def build_vocab(token_lists, special_tokens, max_size=None, min_freq=1):
    '''Return (word_to_id, id_to_word, counts). special_tokens take IDs 0, 1, 2, ... in order;
    the rest are sorted by frequency (ties broken alphabetically so the vocab is reproducible).'''
    counts = Counter(w for tokens in token_lists for w in tokens)
    words = sorted((w for w, c in counts.items() if c >= min_freq and w not in special_tokens),
                   key=lambda w: (-counts[w], w))
    if max_size is not None:
        words = words[:max(0, max_size - len(special_tokens))]
    id_to_word = list(special_tokens) + words
    word_to_id = {w: i for i, w in enumerate(id_to_word)}
    return word_to_id, id_to_word, counts


def encode(tokens, word_to_id):
    return [word_to_id.get(w, UNK_ID) for w in tokens]


def decode(ids, id_to_word, skip_special=True):
    words = [id_to_word[i] if 0 <= i < len(id_to_word) else UNK for i in ids]
    if skip_special:
        words = [w for w in words if w not in (PAD, EOS)]
    return words

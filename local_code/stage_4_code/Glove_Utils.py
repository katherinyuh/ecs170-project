'''
Load pretrained GloVe word vectors for the words in a Stage 4 vocabulary.

GloVe files are plain text, one word per line followed by its numbers:
    the 0.418 0.24968 -0.41242 ...
Only the vocabulary's words are kept, so a 1.7 GB file becomes a small matrix.
'''

from pathlib import Path
import time
import zipfile
import io

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GLOVE_DIR = ROOT / 'data' / 'stage_4_data' / 'glove'


def find_glove_file(dim=100, search_dirs=None):
    '''Return the first GloVe .txt (or .zip holding one) for this dimension, or None.
    Looks in data/stage_4_data/ first, then in your Downloads folder (top level only).'''
    project_dirs = [GLOVE_DIR, ROOT / 'data' / 'stage_4_data', ROOT / 'data']
    for d in [Path(x) for x in (search_dirs or project_dirs)]:
        if not d.is_dir():
            continue
        for pattern in (f'*{dim}d*.txt', f'*{dim}d*.zip'):
            hits = sorted(p for p in d.rglob(pattern) if 'glove' in p.name.lower())
            if hits:
                return hits[0]
    if search_dirs is None:
        downloads = Path.home() / 'Downloads'
        if downloads.is_dir():
            hits = sorted((p for p in downloads.glob(f'*glove*{dim}d*')
                           if p.suffix in ('.zip', '.txt')), key=lambda p: -p.stat().st_mtime)
            if hits:
                return hits[0]
    return None


def _lines(path):
    path = Path(path)
    if path.suffix == '.zip':
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith('.txt'))
            with z.open(name) as raw:
                for line in io.TextIOWrapper(raw, encoding='utf-8', errors='ignore'):
                    yield line
    else:
        with path.open(encoding='utf-8', errors='ignore') as f:
            yield from f


def load_glove_matrix(path, id_to_word, dim=None, seed=2, pad_id=0):
    '''Build an embedding matrix (len(id_to_word), dim) from a GloVe file.

    Words found in GloVe get their GloVe vector. Words not found (and <unk>) get small random
    vectors so they can still be learned. <pad> stays all zeros.
    Returns (matrix, stats).
    '''
    start = time.time()
    wanted = {w: i for i, w in enumerate(id_to_word)}
    found = {}
    for line in _lines(path):
        word, _, rest = line.rstrip().partition(' ')
        if word in wanted and word not in found:
            try:
                vec = np.array(rest.split(), dtype=np.float32)
            except ValueError:
                continue
            if dim is None and len(vec) >= 10:   # skip a header line like "1291147 100"
                dim = len(vec)
            if dim is not None and len(vec) == dim:
                found[word] = vec
    if dim is None:
        raise ValueError(f'No vectors read from {path}')
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0, 0.3, size=(len(id_to_word), dim)).astype(np.float32)
    for word, vec in found.items():
        matrix[wanted[word]] = vec
    matrix[pad_id] = 0.0
    stats = {
        'glove_file': Path(path).name,
        'glove_dim': dim,
        'vocab_words': len(id_to_word),
        'found_in_glove': len(found),
        'coverage': len(found) / max(len(id_to_word) - 2, 1),
        'seconds': round(time.time() - start, 1),
    }
    return matrix, stats

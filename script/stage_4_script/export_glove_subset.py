'''
Person B: save only the GloVe vectors for the IMDB vocabulary into a small file
(data/stage_4_data/glove/glove_subset_100d.txt, about 10 MB instead of 582 MB), so the
same vectors can be used on Colab without uploading the full GloVe zip.
Right-click > Run in PyCharm. Takes under a minute.
'''

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_code.stage_4_code.Dataset_Loader_Classification import Dataset_Loader_Classification
from local_code.stage_4_code.Glove_Utils import _lines

OUT = ROOT / 'data' / 'stage_4_data' / 'glove' / 'glove_subset_100d.txt'


def main():
    downloads = Path.home() / 'Downloads'
    hits = sorted(downloads.glob('*glove*100d*.zip'), key=lambda p: -p.stat().st_mtime)
    if not hits:
        sys.exit('No GloVe 100d zip found in Downloads.')
    vocab = set(Dataset_Loader_Classification().load()['vocab']['id_to_word'])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with OUT.open('w', encoding='utf-8') as f:
        for line in _lines(hits[0]):
            word = line.split(' ', 1)[0]
            if word in vocab and len(line.split()) == 101:
                f.write(line if line.endswith('\n') else line + '\n')
                kept += 1
                vocab.discard(word)
    print(f'Read {hits[0].name}, saved {kept} word vectors to {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)')
    print('Done. Tell Claude it finished.')


if __name__ == '__main__':
    main()

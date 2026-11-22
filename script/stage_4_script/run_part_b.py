'''
Person B: one-click run of the whole classification part (right-click > Run in PyCharm).

1. If data/stage_4_data/ is missing, unzips stage_4_data.zip from your Downloads folder into it
2. Finds the GloVe 100d file (in data/stage_4_data/ or your Downloads) and reads it in place
3. Trains RNN, LSTM and GRU with GloVe embeddings and saves curves, the results table,
   the comparison chart and a log in result/stage_4_result/classification/
'''

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data' / 'stage_4_data'
EPOCHS = 3


def ensure_data():
    if (DATA / 'text_classification' / 'train' / 'pos').is_dir() and (DATA / 'text_generation' / 'data').is_file():
        print('Stage 4 data already in', DATA)
        return
    downloads = Path.home() / 'Downloads'
    zips = sorted(downloads.glob('stage_4_data*.zip'), key=lambda p: -p.stat().st_mtime)
    if not zips:
        sys.exit('Could not find stage_4_data.zip in Downloads. Download the Stage 4 dataset and run this again.')
    print('Unzipping', zips[0], 'into', DATA.parent)
    with zipfile.ZipFile(zips[0]) as z:
        members = [n for n in z.namelist() if n.startswith('stage_4_data/') and '/.DS_Store' not in n]
        z.extractall(DATA.parent, members)
    print('Unzipped', len(members), 'files')


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    ensure_data()
    import torch
    import os
    torch.set_num_threads(os.cpu_count() or 1)
    # no arguments = the full run; any arguments are passed through (e.g. --quick for a test)
    sys.argv = [sys.argv[0]] + (sys.argv[1:] or ['--epochs', str(EPOCHS)])
    import run_classification_experiments
    run_classification_experiments.main()

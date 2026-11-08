'''
Person D helper: finds the instructor's Stage 3 zip in your Downloads folder and
extracts MNIST, ORL and CIFAR into data/stage_3_data using Part A's setup_data.py.
No arguments needed, so it runs with right-click > Run in PyCharm.
'''

import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile, BadZipFile

ROOT = Path(__file__).resolve().parents[2]
NEEDED = {'stage_3_data/MNIST', 'stage_3_data/ORL', 'stage_3_data/CIFAR'}


def find_archive():
    downloads = Path.home() / 'Downloads'
    # check named .zip files anywhere in Downloads, plus any large file at the top level
    # (some browsers save a finished download under a temporary name with no .zip ending)
    candidates = set(downloads.rglob('*.zip'))
    for path in downloads.iterdir():
        try:
            if path.is_file() and path.stat().st_size > 50_000_000:
                candidates.add(path)
        except OSError:
            continue
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            with ZipFile(path) as z:
                if NEEDED <= set(z.namelist()):
                    return path
        except (BadZipFile, OSError):
            continue
    return None


def main():
    data_dir = ROOT / 'data' / 'stage_3_data'
    if all((data_dir / n).is_file() for n in ('MNIST', 'ORL', 'CIFAR')):
        print('Data already extracted in', data_dir)
        return
    archive = find_archive()
    if archive is None:
        print('Could not find the Stage 3 zip in your Downloads folder.')
        print('Download it from the Stage 3 Dataset link, leave it zipped in Downloads, and run this again.')
        sys.exit(1)
    print('Found', archive)
    proc = subprocess.run([sys.executable, str(ROOT / 'script' / 'stage_3_script' / 'setup_data.py'),
                           '--archive', str(archive)], capture_output=True, text=True)
    print(proc.stdout, end='')
    print(proc.stderr, end='')
    if proc.returncode != 0:
        sys.exit(proc.returncode)
    print('Stage 3 data is ready.')


if __name__ == '__main__':
    from run_log import start_log
    start_log('unzip_stage3_data')
    main()

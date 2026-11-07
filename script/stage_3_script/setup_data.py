"""Create Stage 3 folders; optionally extract the instructor's dataset archive."""

import argparse
from pathlib import Path
import shutil
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Path to the provided stage_3_data.zip')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data' / 'stage_3_data')
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    result_dir = ROOT / 'result' / 'stage_3_result'
    result_dir.mkdir(parents=True, exist_ok=True)

    if args.archive:
        with ZipFile(args.archive) as archive:
            # Extract only these exact members, ignoring macOS metadata and other files.
            names = {name: f'stage_3_data/{name}' for name in ('MNIST', 'ORL', 'CIFAR')}
            for member in names.values():
                archive.getinfo(member)
            for name, member in names.items():
                destination = args.data_dir / name
                with archive.open(member) as source, destination.open('wb') as target:
                    shutil.copyfileobj(source, target)
                print(f'Extracted {destination}')

    print(f'Data folder: {args.data_dir.resolve()}')
    print(f'Result folder: {result_dir}')
    missing = [name for name in ('MNIST', 'ORL', 'CIFAR')
               if not (args.data_dir / name).is_file()]
    if missing:
        print('Still needed: ' + ', '.join(missing))
        print('Run again with --archive PATH_TO_stage_3_data.zip to extract the files.')


if __name__ == '__main__':
    main()

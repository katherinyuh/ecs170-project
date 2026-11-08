'''
Copies everything a script prints (including errors) into
result/stage_3_result/logs/<name>.log, so results can be checked later.
'''

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def start_log(name):
    log_dir = ROOT / 'result' / 'stage_3_result' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(log_dir / f'{name}.log', 'w')
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    print(f'(logging to {log_dir / (name + ".log")})')
    return log_file

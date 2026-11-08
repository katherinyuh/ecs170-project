'''
Person D: quick check of all CIFAR experiments on the real data
(2 epochs on 5,000 training images each). Same as
  run_cifar_experiments.py --quick
but runs with right-click > Run in PyCharm.
'''

import sys
from pathlib import Path

sys.argv = [sys.argv[0], '--quick']
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cifar_experiments

run_cifar_experiments.main()

'''
Person D: GPU setup check for Stage 3.
Shows which device the CNNs will use (CUDA on Nvidia, MPS on Apple Silicon Macs,
otherwise CPU) and times a few CIFAR-sized training steps on each available device.

Run from repo root (or press Run in PyCharm):
  python script/stage_3_script/check_device.py
'''

import platform
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR, pick_device


def time_steps(device, steps=20, batch=128):
    model = Method_CNN_CIFAR('timing', '', config={'device_name': str(device)})
    model.build_model(3, 10, (32, 32))
    model.train()
    opt = model.get_optimizer()
    loss_fn = model.get_loss_function()
    x = torch.rand(batch, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (batch,), device=device)

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()

    for _ in range(3):  # warm-up steps are not timed
        loss = loss_fn(model(model._augment(x)), y)
        opt.zero_grad(); loss.backward(); opt.step()
    sync()
    start = time.time()
    for _ in range(steps):
        loss = loss_fn(model(model._augment(x)), y)
        opt.zero_grad(); loss.backward(); opt.step()
    sync()
    per_step = (time.time() - start) / steps
    # one CIFAR epoch is 50,000 / 128 = 391 steps
    return per_step, per_step * 391


def main():
    print('Python      :', sys.version.split()[0])
    print('PyTorch     :', torch.__version__)
    print('Machine     :', platform.platform(), platform.machine())
    print('CUDA (Nvidia GPU) available :', torch.cuda.is_available())
    mps_built = hasattr(torch.backends, 'mps') and torch.backends.mps.is_built()
    mps_ok = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    print('MPS (Apple GPU) built        :', mps_built)
    print('MPS (Apple GPU) available    :', mps_ok)
    chosen = pick_device()
    print('\nThe CNN scripts will train on:', chosen)

    devices = [torch.device('cpu')]
    if chosen.type != 'cpu':
        devices.append(chosen)
    print('\nTiming the baseline CIFAR CNN (batch 128, augmentation on):')
    results = {}
    for device in devices:
        per_step, per_epoch = time_steps(device)
        results[device.type] = per_epoch
        print(f'  {device.type:<4}: {per_step * 1000:7.1f} ms/step  ->  about '
              f'{per_epoch / 60:5.1f} min per epoch, {per_epoch * 20 / 60:6.1f} min for 20 epochs')
    if 'cpu' in results and len(results) > 1:
        other = [k for k in results if k != 'cpu'][0]
        print(f'\n{other.upper()} is about {results["cpu"] / results[other]:.1f}x faster than CPU.')
    elif chosen.type == 'cpu':
        print('\nNo GPU found, so training runs on the CPU. Use --epochs 10 or --quick '
              'in run_cifar_experiments.py to keep runs shorter.')


if __name__ == '__main__':
    from run_log import start_log
    start_log('check_device')
    main()

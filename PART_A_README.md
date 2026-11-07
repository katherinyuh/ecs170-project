# Stage 3: Part A data handoff

This contribution adds the three data loaders, folder setup, validation script,
and a Stage 3 import of the existing Stage 2 pre-split Setting. It uses the actual
instructor-provided files and preserves their training/test partitions and order.

## Install and run

Copy the package contents into the root of the existing `ecs170-project` checkout,
keeping the `local_code/` and `script/` directories in the same locations.
Run these commands from the repository root:

```bash
python -m pip install -r requirements.txt
python script/stage_3_script/setup_data.py --archive /path/to/stage_3_data.zip
python script/stage_3_script/check_data.py --output result/stage_3_result/data_validation.json
```

Download the instructor's archive from the Stage 3 Dataset link in the project
description: https://drive.google.com/file/d/1-udd9jkldM1gEQ-OO-9HKXds42AiLOL6/view

Replace `/path/to/stage_3_data.zip` with the path to that download. The setup script
extracts the three extensionless pickle files into `data/stage_3_data/` and creates
`result/stage_3_result/`. Existing extracted files can be used directly. Both
directories are already excluded from Git by the repository's `.gitignore`.
The zip package contains code and the recorded validation output, not the large
dataset archive. Load pickle data only from the instructor's trusted download.

## Shared interface for the CNN model owners

Every loader returns the same nested dictionary used in Stage 2:

```python
{
    'train': {'X': X_train, 'y': y_train},
    'test': {'X': X_test, 'y': y_test},
}
```

`X` is a contiguous NumPy `float32` array in `(N, C, H, W)` order with pixel
values divided by 255. `y` is a NumPy `int64` vector of class indices. Both are
ready for `torch.from_numpy`. Preprocessing is performed by the loader, so the
CNN code should consume these values directly without another normalization
or image-axis transpose. If copying the Stage 2 MLP's normalization code,
disable that normalization in the CNN.

| Dataset | Train X shape | Test X shape | Labels | Classes |
| --- | --- | --- | --- | --- |
| MNIST | `(60000, 1, 28, 28)` | `(10000, 1, 28, 28)` | `0..9` | 10 |
| ORL | `(360, 1, 112, 92)` | `(40, 1, 112, 92)` | `0..39` | 40 |
| CIFAR | `(50000, 3, 32, 32)` | `(10000, 3, 32, 32)` | `0..9` | 10 |

ORL's three source channels are checked for equality before selecting the first
channel. Original person IDs `1..40` are mapped to class indices `0..39` by
subtracting one in both splits. Add one only when displaying a predicted original
person ID; evaluation compares zero-based predictions against zero-based labels.

Example for the MNIST model owner (substitute ORL or CIFAR as needed):

```python
from local_code.stage_3_code.Dataset_Loader_MNIST import Dataset_Loader_MNIST
from local_code.stage_3_code.Setting_Train_Test_PreSplit import Setting_Train_Test_PreSplit

data_obj = Dataset_Loader_MNIST('MNIST', 'Handwritten digits')
# Defaults to this repository's data/stage_3_data, regardless of working directory.
# Optional override: data_obj.dataset_source_folder_path = '/your/data/folder'

# Once the model, saver and evaluator are supplied by the other team members:
setting_obj = Setting_Train_Test_PreSplit('pre-split train/test', '')
setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
scores = setting_obj.load_run_save_evaluate()
```

The Setting is imported directly from Stage 2, so its interface is unchanged:
it calls `dataset.load()`, assigns `method.data`, calls `method.run()`, saves its
returned dictionary, and evaluates that dictionary. Each CNN's `run()` must
return `{'pred_y': ..., 'true_y': ...}` for the existing saver and metrics.
Use 10 output logits for MNIST/CIFAR and 40 for ORL. Training and test label
indices already match cross-entropy loss conventions.

## Validation

`check_data.py` checks every image's source shape and pixel range, every label,
the complete per-class histograms, and the supplied split sizes. It prints the
output shapes, dtypes, pixel ranges and label counts. Use `--dataset ORL` to
check one dataset, or `--data-dir /your/folder` to override the data location.
It exits with an error if a check fails.

The package's `PART_A_VALIDATION.txt` and `PART_A_VALIDATION.json` record a run
on the instructor's complete archive. `PART_A_VERIFICATION.txt` records separate
pixel/label alignment checks and a Setting integration check without CNN training.

The code was prepared against repository revision
`33af0054ec123ddcb781248a2c182e86e70fd3e0`. Stage 2 files are unchanged.

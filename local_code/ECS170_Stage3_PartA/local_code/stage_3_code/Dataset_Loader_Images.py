"""Shared loader for the instructor's three pre-split Stage 3 pickle files."""

from pathlib import Path
import pickle

import numpy as np

from local_code.base_class.dataset import dataset


DATA_DIRECTORY = Path(__file__).resolve().parents[2] / 'data' / 'stage_3_data'


class Dataset_Loader_Images(dataset):
    # Dataset-specific subclasses supply these values.
    file_name = None
    source_shape = None
    image_shape = None  # (channels, height, width)
    num_classes = None
    label_offset = 0
    expected_label_counts = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName or self.file_name, dDescription or '')
        self.dataset_source_folder_path = DATA_DIRECTORY
        self.data = None

    def _convert_split(self, rows, split):
        expected = np.asarray(self.expected_label_counts[split], dtype=np.int64)
        if len(rows) != int(expected.sum()):
            raise ValueError(
                f'{self.file_name} {split}: expected {expected.sum()} images, '
                f'found {len(rows)}. Use the instructor-provided split.'
            )

        X = np.empty((len(rows), *self.image_shape), dtype=np.float32)
        y = np.empty(len(rows), dtype=np.int64)
        for index, instance in enumerate(rows):
            image = np.asarray(instance['image'])
            label = instance['label']
            if image.shape != self.source_shape:
                raise ValueError(
                    f'{self.file_name} {split}[{index}]: expected image shape '
                    f'{self.source_shape}, found {image.shape}.'
                )
            if not np.isfinite(image).all() or image.min() < 0 or image.max() > 255:
                raise ValueError(f'{self.file_name} {split}[{index}]: invalid pixels.')
            if not isinstance(label, (int, np.integer)):
                raise ValueError(f'{self.file_name} {split}[{index}]: label must be an integer.')
            label = int(label) - self.label_offset
            if not 0 <= label < self.num_classes:
                raise ValueError(f'{self.file_name} {split}[{index}]: label out of range.')

            if self.file_name == 'ORL':
                if not (np.array_equal(image[..., 0], image[..., 1])
                        and np.array_equal(image[..., 0], image[..., 2])):
                    raise ValueError(f'ORL {split}[{index}]: RGB channels are not equal.')
                image = image[..., 0][None, :, :]
            elif image.ndim == 2:
                image = image[None, :, :]
            else:
                image = image.transpose(2, 0, 1)

            X[index] = image
            y[index] = label

        # All three provided files store pixels on the 0..255 scale.
        # Use the same fixed transformation for train and test.
        X /= 255.0
        counts = np.bincount(y, minlength=self.num_classes)
        if not np.array_equal(counts, expected):
            raise ValueError(
                f'{self.file_name} {split}: unexpected label counts: {counts.tolist()}.'
            )
        print(f'{self.file_name} {split}: X={X.shape} ({X.dtype}), y={y.shape} ({y.dtype})')
        print(f'  pixel range: [{X.min():.6f}, {X.max():.6f}]')
        print(f'  label counts: {dict(enumerate(counts.tolist()))}')
        return {'X': X, 'y': y}

    def load(self):
        path = Path(self.dataset_source_folder_path) / self.file_name
        if not path.is_file():
            raise FileNotFoundError(
                f'Missing dataset: {path}. Extract the instructor\'s stage_3_data.zip '
                'with script/stage_3_script/setup_data.py --archive PATH_TO_ZIP.'
            )
        # These are the instructor-provided pickle files, not CSV files.
        with path.open('rb') as stream:
            loaded = pickle.load(stream)
        if not isinstance(loaded, dict) or not {'train', 'test'} <= loaded.keys():
            raise ValueError(f'{path}: expected a dictionary with train and test splits.')
        self.data = {split: self._convert_split(loaded[split], split)
                     for split in ('train', 'test')}
        return self.data

from local_code.stage_3_code.Dataset_Loader_Images import Dataset_Loader_Images


class Dataset_Loader_ORL(Dataset_Loader_Images):
    file_name = 'ORL'
    source_shape = (112, 92, 3)
    image_shape = (1, 112, 92)
    num_classes = 40
    # Original person IDs 1..40 become class indices 0..39 in both splits.
    label_offset = 1
    expected_label_counts = {'train': (9,) * 40, 'test': (1,) * 40}

from local_code.stage_3_code.Dataset_Loader_Images import Dataset_Loader_Images


class Dataset_Loader_CIFAR(Dataset_Loader_Images):
    file_name = 'CIFAR'
    source_shape = (32, 32, 3)
    image_shape = (3, 32, 32)
    num_classes = 10
    expected_label_counts = {'train': (5000,) * 10, 'test': (1000,) * 10}

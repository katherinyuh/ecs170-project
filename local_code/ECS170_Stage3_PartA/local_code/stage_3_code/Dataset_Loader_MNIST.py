from local_code.stage_3_code.Dataset_Loader_Images import Dataset_Loader_Images


class Dataset_Loader_MNIST(Dataset_Loader_Images):
    file_name = 'MNIST'
    source_shape = (28, 28)
    image_shape = (1, 28, 28)
    num_classes = 10
    expected_label_counts = {
        'train': (5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949),
        'test': (980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009),
    }

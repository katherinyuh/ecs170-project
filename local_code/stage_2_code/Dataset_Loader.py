from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load_file(self, file_name):
        X, y = [], []
        with open(self.dataset_source_folder_path + file_name, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                elements = [int(i) for i in line.split(',')]
                y.append(elements[0])    # label is first
                X.append(elements[1:])   # 784 features
        return {'X': X, 'y': y}

    def load(self):
        print('loading data...')
        return {
            'train': self.load_file('train.csv'),
            'test': self.load_file('test.csv'),
        }
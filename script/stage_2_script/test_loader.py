from local_code.stage_2_code.Dataset_Loader import Dataset_Loader

data_obj = Dataset_Loader('stage 2', '')
data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
data = data_obj.load()

for split in ['train', 'test']:
    X, y = data[split]['X'], data[split]['y']
    print(split, '-> rows:', len(X), '| features per row:', len(X[0]),
          '| labels:', sorted(set(y)))
from local_code.base_class.dataset import dataset
import os
import pickle


class Dataset_Loader_Classification(dataset):
    dataset_source_file_path = None

    def __init__(self, dName=None, dDescription=None):
        dataset.__init__(self, dName, dDescription)

    def load(self):
        if self.dataset_source_file_path is None:
            raise ValueError('dataset_source_file_path is not set')
        path = os.path.normpath(self.dataset_source_file_path)
        print('loading classification cache:', path)
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print('train size:', len(data['train']['y']))
        print('test size:', len(data['test']['y']))
        print('sequence length:', data['train']['X'].shape[1])
        print('vocab size:', data['vocab_size'])
        return data

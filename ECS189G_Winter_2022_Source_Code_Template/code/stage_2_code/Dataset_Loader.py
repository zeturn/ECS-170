'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import os
import numpy as np


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    train_dataset_source_file_name = None
    test_dataset_source_file_name = None
    delimiter = None
    label_column_index = 0
    normalize_feature = True
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def _load_feature_label_file(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError('Cannot find dataset file: ' + file_path)

        rows = []
        with open(file_path, 'r') as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                if self.delimiter is None:
                    elements = line.replace(',', ' ').split()
                else:
                    elements = line.split(self.delimiter)
                rows.append([float(i) for i in elements])

        if len(rows) == 0:
            raise ValueError('Dataset file is empty: ' + file_path)

        array_data = np.array(rows, dtype=np.float32)

        if self.label_column_index == 0:
            y = array_data[:, 0].astype(int)
            X = array_data[:, 1:]
        elif self.label_column_index == -1:
            y = array_data[:, -1].astype(int)
            X = array_data[:, :-1]
        else:
            y = array_data[:, self.label_column_index].astype(int)
            X = np.delete(array_data, self.label_column_index, axis=1)

        if self.normalize_feature:
            X = X / 255.0
        return X, y
    
    def load(self):
        print('loading data...')
        if self.dataset_source_folder_path is None:
            raise ValueError('dataset_source_folder_path is required for Stage 2 dataset loading.')
        if self.train_dataset_source_file_name is None or self.test_dataset_source_file_name is None:
            raise ValueError('Both train_dataset_source_file_name and test_dataset_source_file_name are required.')

        train_path = os.path.join(self.dataset_source_folder_path, self.train_dataset_source_file_name)
        test_path = os.path.join(self.dataset_source_folder_path, self.test_dataset_source_file_name)

        train_X, train_y = self._load_feature_label_file(train_path)
        test_X, test_y = self._load_feature_label_file(test_path)

        return {
            'train': {'X': train_X, 'y': train_y},
            'test': {'X': test_X, 'y': test_y}
        }
from local_code.base_class.dataset import dataset
import os
import pickle
import numpy as np


class Dataset_Loader(dataset):
    dataset_source_folder_path = None
    dataset_source_file_name = None
    image_size = (28, 28)
    convert_gray = True
    normalize = True

    def __init__(self, dName=None, dDescription=None):
        dataset.__init__(self, dName, dDescription)

    def load(self):
        print(f'loading {self.dataset_name} data...')

        path = os.path.join(
            self.dataset_source_folder_path,
            self.dataset_source_file_name
        )

        path = os.path.normpath(path)
        print('dataset path:', path)

        with open(path, 'rb') as f:
            raw_data = pickle.load(f)

        def preprocess_data(instances):
            X, y = [], []
            for instance in instances:
                image = np.array(instance['image'])
                if len(image.shape) == 3 and image.shape[2] == 3:
                    if not self.keep_rgb:
                        image = image.mean(axis=2)  # 原本是[:,:,0]但是说这样是只取R通道，改了是grayscale
                # float
                image = image.astype(np.float32)
                # normalize
                if self.normalize and image.max() > 1.0:
                    image /= 255.0

                X.append(image)
                y.append(int(instance['label']))
            return np.stack(X), np.array(y)
        train_X, train_y = preprocess_data(raw_data['train'])
        test_X, test_y = preprocess_data(raw_data['test'])
        # --- 通用性处理 3: 标签归零对齐 ---
        # 无论最小标签是 1 还是 0，都统一减去最小值，确保从 0 开始
        min_label = min(train_y.min(), test_y.min())
        if min_label != 0:
            print(f"Adjusting labels by subtracting {min_label} for 0-indexing...")
            train_y = train_y - min_label
            test_y = test_y - min_label
        print('train size:', len(train_y))
        print('test size:', len(test_y))
        print('image shape:', train_X[0].shape)
        print("DEBUG X shape:", train_X.shape)
        print("DEBUG sample shape:", train_X[0].shape)
        print("DEBUG min/max:", train_X[0].min(), train_X[0].max())
        return {
            'train': {'X': train_X, 'y': train_y},
            'test': {'X': test_X, 'y': test_y}
        }

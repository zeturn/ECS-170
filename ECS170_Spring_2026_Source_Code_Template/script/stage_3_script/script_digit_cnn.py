import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)
os.chdir(current_dir)

from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN import Method_CNN
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch


if __name__ == '__main__':
    np.random.seed(2)
    torch.manual_seed(2)

    DATASET_NAME = 'CIFAR'  # 'ORL' / 'MNIST' / 'CIFAR'

    DATASET_CONFIG = {
        'ORL': {
            'task_name': 'ORL',
            'image_size': (112, 92),
            'epochs': 35
        },
        'MNIST': {
            'task_name': 'MNIST',
            'image_size': (28, 28),
            'epochs': 20
        },
        'CIFAR': {
            'task_name': 'CIFAR',
            'image_size': (32, 32),
            'epochs': 35
        }
    }

    config = DATASET_CONFIG[DATASET_NAME]

    TASK_NAME = config['task_name']
    EXPECTED_IMAGE_SIZE = config['image_size']
    EPOCHS = config['epochs']
    #parameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001

    print(f'************ Start Stage 3: {TASK_NAME} CNN ************')

    # Dataset
    data_obj = Dataset_Loader(f'{DATASET_NAME} dataset', '')
    data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
    data_obj.dataset_source_file_name = DATASET_NAME
    data_obj.image_size = EXPECTED_IMAGE_SIZE
    data_obj.convert_gray = True
    data_obj.normalize = True

    # CIFAR use RGB, others grayscale
    if DATASET_NAME == 'CIFAR':
        data_obj.keep_rgb = True
    else:
        data_obj.keep_rgb = False

    # Method
    method_obj = Method_CNN('convolutional neural network', '')
    method_obj.max_epoch = EPOCHS
    method_obj.learning_rate = LEARNING_RATE
    method_obj.batch_size = BATCH_SIZE

    # independent folder for each task
    method_obj.history_destination_folder_path = f'../../result/stage_3_result/{DATASET_NAME}/'

    # Result Saver
    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = f'../../result/stage_3_result/{DATASET_NAME}/'

    # file name
    result_obj.result_destination_file_name = f'{DATASET_NAME.lower()}_cnn_result'

    # Setting
    setting_obj = Setting_Train_Test_Split('train test split', '')
    setting_obj.test_size = 0.2
    setting_obj.random_state = 2

    evaluate_obj = Evaluate_Accuracy('accuracy', '')

    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()

    score, _ = setting_obj.load_run_save_evaluate()

    print('************ Overall Performance ************')
    print(f'{TASK_NAME} CNN Accuracy: {score}')
    print('************ Finish ************')

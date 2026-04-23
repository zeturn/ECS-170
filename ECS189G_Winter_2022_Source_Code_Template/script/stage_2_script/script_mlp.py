import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from code.stage_2_code.Dataset_Loader import Dataset_Loader
from code.stage_2_code.Method_MLP import Method_MLP
from code.stage_2_code.Result_Saver import Result_Saver
from code.stage_2_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy


def save_learning_curve(learning_curve, save_path, title):
    epochs = learning_curve['epoch']
    train_loss = learning_curve['train_loss']
    train_accuracy = learning_curve['train_accuracy']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, color='tab:red')
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')

    axes[1].plot(epochs, train_accuracy, color='tab:blue')
    axes[1].set_title('Training Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def run_experiment(config_name, method_kwargs):
    data_obj = Dataset_Loader('stage_2_dataset', '')
    data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    data_obj.train_dataset_source_file_name = 'train.csv'
    data_obj.test_dataset_source_file_name = 'test.csv'
    data_obj.delimiter = ','
    data_obj.label_column_index = 0
    data_obj.normalize_feature = True

    method_obj = Method_MLP('multi-layer perceptron', '', **method_kwargs)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_' + config_name + '_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test_Split('predefined train test split', '')
    evaluate_obj = Evaluate_Accuracy('multiclass metrics', '')

    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    metrics, learned_result = setting_obj.load_run_save_evaluate()

    learning_curve_path = '../../result/stage_2_result/learning_curve_' + config_name + '.png'
    save_learning_curve(learned_result['learning_curve'], learning_curve_path, 'MLP Learning Curve - ' + config_name)

    return metrics, learning_curve_path


if __name__ == '__main__':
    np.random.seed(2)
    torch.manual_seed(2)

    # Stage 2 requests trying model improvements.
    experiment_settings = [
        {
            'name': 'baseline',
            'kwargs': {
                'hidden_dims': [128, 64],
                'max_epoch': 20,
                'learning_rate': 1e-3,
                'batch_size': 128,
                'dropout': 0.1,
                'optimizer_name': 'adam',
                'loss_name': 'cross_entropy',
                'weight_decay': 0.0,
                'verbose_step': 5
            }
        },
        {
            'name': 'improved',
            'kwargs': {
                'hidden_dims': [256, 128, 64],
                'max_epoch': 30,
                'learning_rate': 5e-4,
                'batch_size': 128,
                'dropout': 0.2,
                'optimizer_name': 'adamw',
                'loss_name': 'cross_entropy_label_smoothing',
                'weight_decay': 1e-4,
                'verbose_step': 5
            }
        }
    ]

    all_results = []
    print('************ Start Stage 2 MLP ************')

    for setting in experiment_settings:
        print('************ Experiment:', setting['name'], '************')
        metrics, curve_path = run_experiment(setting['name'], setting['kwargs'])
        all_results.append({'name': setting['name'], 'metrics': metrics, 'curve_path': curve_path})
        print('Metrics:', metrics)
        print('Learning Curve:', curve_path)

    best_result = max(all_results, key=lambda item: item['metrics']['macro_f1'])

    summary_path = '../../result/stage_2_result/mlp_stage2_summary.txt'
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        for result in all_results:
            f.write('Experiment: ' + result['name'] + '\n')
            for metric_name, metric_value in result['metrics'].items():
                f.write(metric_name + ': ' + str(metric_value) + '\n')
            f.write('learning_curve: ' + result['curve_path'] + '\n\n')
        f.write('Best model by macro_f1: ' + best_result['name'] + '\n')

    print('************ Final Report ************')
    print('Best experiment:', best_result['name'])
    print('Best metrics:', best_result['metrics'])
    print('Summary saved to:', summary_path)
    print('************ Finish ************')

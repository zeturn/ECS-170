import argparse
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
repo_root = os.path.abspath(os.path.join(project_root, '..'))
sys.path.insert(0, project_root)
os.chdir(current_dir)

import numpy as np
import torch

from local_code.stage_4_code.Dataset_Loader_Classification import Dataset_Loader_Classification
from local_code.stage_4_code.Method_RNN_Classifier import Method_RNN_Classifier
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Setting_Train_Test_Split import Setting_Train_Test_Split


def parse_args():
    parser = argparse.ArgumentParser(description='Stage 4 text classification with RNN/LSTM/GRU.')
    parser.add_argument('--model', choices=['rnn', 'lstm', 'gru', 'all'], default='all')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--embed-dim', type=int, default=128)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument(
        '--cache',
        default=os.path.join(
            repo_root,
            'stage_4_data',
            'stage_4_data',
            'text_classification',
            'processed_classification_cache_len128.pkl',
        ),
    )
    return parser.parse_args()


def run_one_model(args, model_name):
    data_obj = Dataset_Loader_Classification('IMDb sentiment classification', '')
    data_obj.dataset_source_file_path = args.cache

    method_obj = Method_RNN_Classifier(f'{model_name} classifier', '')
    method_obj.cell_type = model_name
    method_obj.max_epoch = args.epochs
    method_obj.batch_size = args.batch_size
    method_obj.learning_rate = args.lr
    method_obj.hidden_dim = args.hidden_dim
    method_obj.embed_dim = args.embed_dim
    method_obj.history_destination_folder_path = f'../../result/stage_4_result/classification/{model_name}/'

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = f'../../result/stage_4_result/classification/{model_name}/'
    result_obj.result_destination_file_name = f'{model_name}_classification_result'

    setting_obj = Setting_Train_Test_Split('train test split', '')
    evaluate_obj = Evaluate_Accuracy('accuracy', '')

    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    evaluation, _ = setting_obj.load_run_save_evaluate()
    score = evaluation['accuracy'] if isinstance(evaluation, dict) else evaluation
    print(f'{model_name.upper()} test accuracy: {score:.4f}')
    if isinstance(evaluation, dict):
        print(
            f"{model_name.upper()} weighted_f1: {evaluation['weighted_f1']:.4f} "
            f"macro_f1: {evaluation['macro_f1']:.4f}"
        )
    return score


if __name__ == '__main__':
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print('************ Start Stage 4 Text Classification ************')
    model_names = ['rnn', 'lstm', 'gru'] if args.model == 'all' else [args.model]
    scores = {}
    for model_name in model_names:
        scores[model_name] = run_one_model(args, model_name)

    print('************ Overall Performance ************')
    for model_name, score in scores.items():
        print(f'{model_name.upper()} Accuracy: {score:.4f}')
    best_model = max(scores, key=scores.get)
    print(f'Best model: {best_model.upper()} ({scores[best_model]:.4f})')
    print('************ Finish ************')

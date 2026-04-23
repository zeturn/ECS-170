'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):
        print('evaluating performance...')
        true_y = np.array(self.data['true_y'])
        pred_y = np.array(self.data['pred_y'])

        evaluation_result = {
            'accuracy': accuracy_score(true_y, pred_y)
        }

        for avg in ['macro', 'micro', 'weighted']:
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_y,
                pred_y,
                average=avg,
                zero_division=0
            )
            evaluation_result[avg + '_precision'] = precision
            evaluation_result[avg + '_recall'] = recall
            evaluation_result[avg + '_f1'] = f1

        return evaluation_result
        
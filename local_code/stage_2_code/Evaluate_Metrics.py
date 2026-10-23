'''
Composite multiclass metrics for Stage 2.

Expects self.data = {'true_y': ..., 'pred_y': ...} (same shape as Method_MLP.run()).
Returns a dict with accuracy plus F1 / precision / recall for macro, micro, weighted.
'''

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


class Evaluate_Metrics(evaluate):
    data = None

    def evaluate(self):
        print('evaluating performance...')
        true_y = self.data['true_y']
        pred_y = self.data['pred_y']

        return {
            'accuracy': accuracy_score(true_y, pred_y),
            'f1_macro': f1_score(true_y, pred_y, average='macro'),
            'f1_micro': f1_score(true_y, pred_y, average='micro'),
            'f1_weighted': f1_score(true_y, pred_y, average='weighted'),
            'precision_macro': precision_score(true_y, pred_y, average='macro', zero_division=0),
            'precision_micro': precision_score(true_y, pred_y, average='micro', zero_division=0),
            'precision_weighted': precision_score(true_y, pred_y, average='weighted', zero_division=0),
            'recall_macro': recall_score(true_y, pred_y, average='macro', zero_division=0),
            'recall_micro': recall_score(true_y, pred_y, average='micro', zero_division=0),
            'recall_weighted': recall_score(true_y, pred_y, average='weighted', zero_division=0),
        }

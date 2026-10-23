'''
Smoke test for Stage 2 metrics using fake true_y / pred_y.
Run from repo root:
  PYTHONPATH=. python script/stage_2_script/test_metrics.py
'''

from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics


def main():
    # Multiclass fake labels (3 classes); not perfect predictions on purpose
    true_y = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    pred_y = [0, 1, 1, 0, 2, 2, 0, 1, 0, 1]

    data = {'true_y': true_y, 'pred_y': pred_y}

    acc = Evaluate_Accuracy('accuracy', 'smoke test')
    acc.data = data
    print('Evaluate_Accuracy:', acc.evaluate())

    metrics = Evaluate_Metrics('metrics', 'smoke test')
    metrics.data = data
    result = metrics.evaluate()
    print('Evaluate_Metrics:')
    for key, value in result.items():
        print(f'  {key}: {value:.4f}')


if __name__ == '__main__':
    main()

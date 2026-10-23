
from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_2_code.Setting_Train_Test_PreSplit import Setting_Train_Test_PreSplit
from local_code.stage_1_code.Result_Saver import Result_Saver
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('MNIST', 'Handwritten digits')
    data_obj.dataset_source_folder_path = '../../data/stage_2_data/'

    method_obj = Method_MLP('multi-layer perceptron', '')
    method_obj.max_epoch = 30
    method_obj.learning_rate = 1e-3
    method_obj.hidden_dims = [256, 128]
    method_obj.activation = 'relu'
    method_obj.optimizer_name = 'adam'
    method_obj.loss_name = 'cross_entropy'
    method_obj.batch_size = 256
    method_obj.print_every = 5
    method_obj.normalize_input = True

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'
    result_obj.fold_count = None

    setting_obj = Setting_Train_Test_PreSplit('pre-split train/test', '')
    evaluate_obj = Evaluate_Metrics('multiclass metrics', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    scores = setting_obj.load_run_save_evaluate()

    print('************ Test Performance ************')
    for name, value in scores.items():
        print(f'{name}: {value:.4f}')

    epochs = np.arange(1, len(method_obj.loss_history) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, method_obj.loss_history)
    axes[0].set(title='Training Loss', xlabel='Epoch', ylabel='Loss')

    axes[1].plot(epochs, method_obj.train_acc_history)
    axes[1].set(
        title='Training Accuracy',
        xlabel='Epoch',
        ylabel='Accuracy',
        ylim=(0, 1.05)
    )

    for ax in axes:
        ax.grid(alpha=0.3)

    fig.tight_layout()
    output_dir = '../../result/stage_2_result/'
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_dir + 'MLP_learning_curves.png', dpi=150)
    plt.close(fig)

    print('************ Finish ************')
    # ------------------------------------------------------

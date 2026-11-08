from local_code.stage_3_code.Dataset_Loader_MNIST import Dataset_Loader_MNIST
from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST

data = Dataset_Loader_MNIST('MNIST', '').load()

model = Method_CNN_MNIST('CNN MNIST', '')
model.max_epoch = 2  # quick check; the real run uses 10
model.data = data
result = model.run()

accuracy = (result['pred_y'] == result['true_y']).mean()
print('Test accuracy:', accuracy)
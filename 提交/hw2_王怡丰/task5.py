import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

filepath = './morpho_mnist'
test_label = np.load(filepath + '/test/test_label.npy')
test_mnist = np.load(filepath + '/test/test_mnist.npy')
train_label = np.load(filepath + '/train/train_label.npy')
train_mnist = np.load(filepath + '/train/train_mnist.npy')


# 通过十折法选择超参数C
def select_C_value_10(C_value):
    global train_label
    global train_mnist
    num = 0
    for i in range(10):
        new_train_mnist = np.delete(train_mnist, np.arange((i * 100), (i * 100 + 100)), axis = 0)
        new_train_mnist_flat = new_train_mnist.reshape(new_train_mnist.shape[0], -1)
        new_train_label = np.delete(train_label, np.arange((i * 100), (i * 100 + 100)), axis = 0)
        new_test_mnist = train_mnist[(i * 100) : (i * 100 + 100), :]
        new_test_mnist_flat = new_test_mnist.reshape(new_test_mnist.shape[0], -1)
        new_test_label = train_label[(i * 100) : (i * 100 + 100)]
        clf = SVC(kernel='linear', C = C_value)
        clf.fit(new_train_mnist_flat, new_train_label)
        predictions_result = clf.predict(new_test_mnist_flat)
        predictions_fit = np.where(predictions_result == new_test_label, 1, 0)
        num += np.sum(predictions_fit)
    return num

# 先大致确定C的位数
# for i in range(-10, 3):
#     C_value = 10 ** i
#     print("C=", C_value)
#     print("正确率:", select_C_value_10(C_value) / 1000)



# 进一步精细化
# for i in range(1, 11):
#     C_value = i * 1e-5
#     print("C=", C_value)
#     print("正确率:", select_C_value_10(C_value) / 1000)

# 确定C_value=1e-4

# 计算最终结果

C_value = 1e-4
train_mnist_flat = train_mnist.reshape(train_mnist.shape[0], -1)
test_mnist_flat = test_mnist.reshape(test_mnist.shape[0], -1)
clf = SVC(kernel='linear', C = C_value)
clf.fit(train_mnist_flat, train_label)
predictions_result = clf.predict(test_mnist_flat)
predictions_fit = np.where(predictions_result == test_label, 1, 0)
print("C=", C_value)
print("正确率:", np.sum(predictions_fit) / 1000)


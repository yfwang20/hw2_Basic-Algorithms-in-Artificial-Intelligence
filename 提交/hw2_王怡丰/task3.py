import numpy as np

filepath = './morpho_mnist'
test_label = np.load(filepath + '/test/test_label.npy')
test_mnist = np.load(filepath + '/test/test_mnist.npy')
train_label = np.load(filepath + '/train/train_label.npy')
train_mnist = np.load(filepath + '/train/train_mnist.npy')

print(np.shape(test_mnist))
import numpy as np

# sigmoid函数，将线性回归的输出映射到0和1之间
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Ridge回归模型的损失函数
def ridge_log_likelihood(X, y, weights, lambd):
    z = np.dot(X, weights)
    epsilon = 1e-10
    log_likelihood = np.sum(y * np.log(sigmoid(z) + epsilon) + (1 - y) * np.log(1 - sigmoid(z) + epsilon))
    ridge_penalty = lambd * np.sum(weights[1:] ** 2)
    return -log_likelihood + ridge_penalty

# 梯度下降法来训练Ridge回归模型
def train(X, y, learning_rate=0.01, num_iterations=2000, lambd = 0.01):
    # 将X中的元素展平
    num_samples = X.shape[0]
    X_flat = X.reshape(num_samples, -1)

    # 添加偏置项（截距项）
    X_bias = np.c_[np.ones(X_flat.shape[0]), X_flat]

    
    # 初始化权重
    weights = np.zeros(X_bias.shape[1])
    
    # 梯度下降循环
    for i in range(num_iterations):
        # 计算预测值
        z = np.dot(X_bias, weights)
        # 计算梯度
        gradient = np.dot(X_bias.T, (sigmoid(z) - y)) / y.size + lambd * np.r_[0, weights[1:]]
        # 更新权重
        weights -= learning_rate * gradient
        
        # # 每隔100次迭代输出损失值
        # if i % 100 == 0:
        #     print(f'Iteration {i}, Loss: {ridge_log_likelihood(X_bias, y, weights, lambd)}')
    
    return weights

# Ridge回归模型的预测函数，输出为1的概率
def predict(X, weights):
    z = np.dot(X, weights)
    return sigmoid(z)


# 通过十折法选择超参数lambda
def select_lambda_10(lambd):
    global train_label
    global train_mnist
    num = 0
    for i in range(10):
        new_train_mnist = np.delete(train_mnist, np.arange((i * 100), (i * 100 + 100)), axis = 0)
        new_train_label = np.delete(train_label, np.arange((i * 100), (i * 100 + 100)), axis = 0)
        new_test_mnist = train_mnist[(i * 100) : (i * 100 + 100), :]
        new_test_label = train_label[(i * 100) : (i * 100 + 100)]
        weights = train(new_train_mnist, new_train_label, lambd)
        new_test_mnist_flat = new_test_mnist.reshape(new_test_mnist.shape[0], -1)
        new_test_mnist_bias = np.c_[np.ones(new_test_mnist_flat.shape[0]), new_test_mnist_flat]
        predictions = predict(new_test_mnist_bias, weights)
        predictions_result = np.where(predictions > 0.5, 1, 0)
        predictions_fit = np.where(predictions_result == new_test_label, 1, 0)
        num += np.sum(predictions_fit)
    return num

# 先大致确定lambd的位数
# for i in range(-6, 1):
#     lambd = 10 ** i
#     print("lambd=", lambd)
#     print("正确率:", select_lambda_10(lambd) / 1000)

# 再精细化lambd的数值
# lambd = 5e-5
# print("lambd=", lambd)
# print("正确率:", select_lambda_10(lambd) / 1000)
# lambd = 5e-4
# print("lambd=", lambd)
# print("正确率:", select_lambda_10(lambd) / 1000)

# 进一步精细化
# for i in range(1, 11):
#     lambd = i * 1e-5
#     print("lambd=", lambd)
#     print("正确率:", select_lambda_10(lambd) / 1000)

# 确定lambd=8e-05

# 计算最终结果

lambd = 8e-05
weights = train(train_mnist, train_label, lambd)
test_mnist_flat = test_mnist.reshape(test_mnist.shape[0], -1)
test_mnist_bias = np.c_[np.ones(test_mnist_flat.shape[0]), test_mnist_flat]
predictions = predict(test_mnist_bias, weights)
predictions_result = np.where(predictions > 0.5, 1, 0)
predictions_fit = np.where(predictions_result == test_label, 1, 0)
print("lambd=", lambd)
print("正确率:", np.sum(predictions_fit) / 1000)


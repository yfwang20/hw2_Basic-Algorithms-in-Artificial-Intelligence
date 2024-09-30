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

# 逻辑回归模型的损失函数（对数似然损失）
def log_likelihood(X, y, weights):
    z = np.dot(X, weights)
    #return np.sum(y * np.log(sigmoid(z)) + (1 - y) * np.log(1 - sigmoid(z)))
    epsilon = 1e-10
    return np.sum(y * np.log(sigmoid(z) + epsilon) + (1 - y) * np.log(1 - sigmoid(z) + epsilon))

# 梯度下降法来训练逻辑回归模型
def train(X, y, learning_rate=0.01, num_iterations=1000):
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
        gradient = np.dot(X_bias.T, (sigmoid(z) - y)) / y.size
        # 更新权重
        weights -= learning_rate * gradient
        
        # 每隔100次迭代输出损失值
        if i % 100 == 0:
            print(f'Iteration {i}, Loss: {log_likelihood(X_bias, y, weights)}')
    
    return weights

# 逻辑回归模型的预测函数，输出为1的概率
def predict(X, weights):
    z = np.dot(X, weights)
    return sigmoid(z)

# 示例数据集（这里仅作示例，实际应用中需要真实数据）
X = train_mnist
y = train_label

# 训练模型
weights = train(X, y)

# 打印训练得到的权重
print("Trained weights:", weights)

# 使用训练好的模型进行预测
test_mnist_flat = test_mnist.reshape(X.shape[0], -1)
test_mnist_bias = np.c_[np.ones(test_mnist_flat.shape[0]), test_mnist_flat]
predictions = predict(test_mnist_bias, weights)
predictions_result = np.where(predictions > 0.5, 1, 0)
predictions_fit = np.where(predictions_result == test_label, 1, 0)
print("Predictions_fit:", np.sum(predictions_fit))
print("正确率:", np.sum(predictions_fit) / 1000)
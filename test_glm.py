import numpy as np

# sigmoid函数，将线性回归的输出映射到0和1之间
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 逻辑回归模型的预测函数
def predict(X, weights):
    z = np.dot(X, weights)
    return sigmoid(z)

# 逻辑回归模型的损失函数（对数似然损失）
def log_likelihood(X, y, weights):
    z = np.dot(X, weights)
    return np.sum(y * np.log(sigmoid(z)) + (1 - y) * np.log(1 - sigmoid(z)))

# 梯度下降法来训练逻辑回归模型
def train(X, y, learning_rate=0.01, num_iterations=1000):
    # 将每个[28,28]的矩阵展平为一个长度为784的向量
    num_samples = X.shape[0]
    X_flat = X.reshape(num_samples, -1)  # 展平操作
    
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

# 示例数据集（这里仅作示例，实际应用中需要真实数据）
# 假设X是一个三维数组，其中每个元素都是一个[28,28]的矩阵
X = np.random.rand(12, 28, 28)  # 12个样本，每个样本是一个[28,28]的矩阵
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1])
print(np.shape(X))
# 训练模型
weights = train(X, y)

# 打印训练得到的权重
print("Trained weights:", weights)

# 使用训练好的模型进行预测
# 在预测前，需要将输入数据展平
X_flat = X.reshape(X.shape[0], -1)
X_bias = np.c_[np.ones(X_flat.shape[0]), X_flat]
predictions = predict(X_bias, weights)

print("Predictions:", predictions)

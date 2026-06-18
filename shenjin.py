import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

print("=== 实验三：Fashion-MNIST 神经网络分类（离线版） ===")

# 1. 造一个和Fashion-MNIST结构完全一样的假数据（不用联网下载）
print("创建模拟数据...")
np.random.seed(42)  # 固定随机种子，保证每次结果一致
n_samples = 10000
n_features = 784  # 28x28像素
n_classes = 10

# 生成随机图像数据和标签
X = np.random.rand(n_samples, n_features)
y = np.random.randint(0, n_classes, size=n_samples)

# 归一化（和真实Fashion-MNIST一样）
X = X / 255.0

# 划分训练集/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✅ 数据准备完成！训练集：{len(X_train)}，测试集：{len(X_test)}")

# 2. 搭建三层全连接神经网络（和实验要求一致）
print("开始训练模型...")
model = MLPClassifier(
    hidden_layer_sizes=(128,),  # 隐藏层128个节点
    activation='logistic',      # 对应sigmoid激活
    solver='adam',
    max_iter=20,                # 训练轮次
    batch_size=256,
    random_state=42,
    verbose=True
)

# 训练模型
model.fit(X_train, y_train)

# 3. 模型评估
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ 测试集准确率：{acc:.4f}")

# 4. 绘制损失曲线（和实验要求一致）
plt.figure(figsize=(10, 4))
plt.plot(model.loss_curve_, label='训练损失')
plt.xlabel("迭代轮次 Epochs")
plt.ylabel("损失 Loss")
plt.title("模型训练损失曲线")
plt.legend()
plt.grid(True)
plt.show()

print("\n=== 实验三运行完成！所有步骤均已实现 ===")
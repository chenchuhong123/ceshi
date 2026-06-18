import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# 1. 加载数据集
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# 2. 数据预处理：归一化 + 展平为二维
x_train = x_train / 255.0
x_test = x_test / 255.0
# 三维(样本,28,28) → 二维(样本,784)
x_train = x_train.reshape(-1, 28*28).astype(np.float32)
x_test = x_test.reshape(-1, 28*28).astype(np.float32)

# 标签独热编码
y_train_onehot = tf.one_hot(y_train, 10)

# 可选：缩小训练集（无GPU时使用）
# x_train = x_train[:10000]
# y_train_onehot = y_train_onehot[:10000]

# 3. 搭建三层全连接神经网络
num_features = 28 * 28
model = keras.Sequential([
    keras.layers.Dense(128, activation='sigmoid', input_shape=(num_features,)),  # 隐藏层
    keras.layers.Dense(10, activation='softmax')  # 输出层 10分类
])

# 4. 模型编译
model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.001),
    loss=tf.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

# 查看网络结构
model.summary()

# 5. 分批训练
epochs = 20
batch_size = 256
history = model.fit(
    x_train, y_train_onehot,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1  # 划分10%训练集为验证集
)

# 6. 绘制损失函数曲线
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.xlabel("迭代轮次 Epochs")
plt.ylabel("损失 Loss")
plt.title("模型训练损失曲线")
plt.legend()
plt.show()

# 7. 测试集评估
test_loss, test_acc = model.evaluate(x_test, tf.one_hot(y_test, 10))
print(f"测试集损失：{test_loss:.4f}，测试集准确率：{test_acc:.4f}")

# ========== 选做：保存模型 ==========
# model.save("fashion_mnist_model.h5")
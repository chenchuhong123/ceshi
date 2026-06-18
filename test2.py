import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("==== 实验二：电离层数据集 PCA+KNN（输出优化版） ====")

# 模拟数据
np.random.seed(42)
X = np.random.rand(351, 34)
y = np.random.randint(0, 2, size=351)

# 划分训练集/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print(f"✅ 数据划分完成：")
print(f"   训练集样本数：{len(X_train)}")
print(f"   测试集样本数：{len(X_test)}")

# PCA降维
pca = PCA()
pca.fit(X_train)
cum_variance = np.cumsum(pca.explained_variance_ratio_)

# 找累计方差首次大于0.8的维度
n = np.argmax(cum_variance >= 0.8) + 1
print(f"\n✅ 累计方差贡献率首次大于0.8的维度：n = {n}")

# 降维
pca_reduce = PCA(n_components=n)
X_train_reduced = pca_reduce.fit_transform(X_train)
X_test_reduced = pca_reduce.transform(X_test)

# KNN分类
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_reduced, y_train)
y_pred = knn.predict(X_test_reduced)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ PCA降维后，KNN测试集准确率：{acc:.4f}")
print("\n==== 实验二运行完成！ ====")
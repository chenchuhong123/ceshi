# ===================== 实验1 导入包（无冗余） =====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===================== 1. 加载数据 =====================
wine = load_wine()
X = wine.data
y = wine.target

print("数据形状：", X.shape)
print("各类样本数量：", np.bincount(y))

# ===================== 2. 划分训练集/测试集 7:3 =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=2022
)

# ===================== 3. 训练随机森林 =====================
model = RandomForestClassifier(random_state=2022)
model.fit(X_train, y_train)

# ===================== 4. 预测与评估 =====================
y_pred = model.predict(X_test)

print("="*50)
print("准确率：", accuracy_score(y_test, y_pred))
print("="*50)
print("分类报告：")
print(classification_report(y_test, y_pred))
print("="*50)
print("混淆矩阵：")
print(confusion_matrix(y_test, y_pred))

# ===================== 5. 超参数搜索 =====================
param_grid = {
    'n_estimators': [20, 40, 60, 80, 100, 120],
    'min_samples_leaf': [1, 2, 3, 4, 5]
}

grid = GridSearchCV(RandomForestClassifier(random_state=2022), param_grid, cv=5)
grid.fit(X_train, y_train)

print("最优参数：", grid.best_params_)
print("最优交叉验证准确率：", grid.best_score_)

# ===================== 6. 热力图 =====================
results = pd.DataFrame(grid.cv_results_)
heat_data = results.pivot(
    index='param_min_samples_leaf',
    columns='param_n_estimators',
    values='mean_test_score'
)

plt.figure(figsize=(10, 4))
sns.heatmap(heat_data, annot=True, fmt='.4f', cmap='YlGnBu')
plt.title('参数热力图')
plt.show()
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from skimage.feature import local_binary_pattern

# ========== 1. 路径设置（你改成自己的即可） ==========
src_path = r"F:\potato\image"  # 五类文件夹所在父目录
classes = ["健康", "早疫病一般", "早疫病严重", "晚疫病一般", "晚疫病严重"]

# ========== 2. 特征提取：LAB颜色 + LBP纹理 ==========
def extract_features(img_path):
    # 读取图片（支持中文路径）
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    img = cv2.resize(img, (200, 200))

    # --- LAB颜色特征：a、b各64维 ---
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    hist_a, _ = np.histogram(a, bins=64, range=(1, 256), density=True)
    hist_b, _ = np.histogram(b, bins=64, range=(1, 256), density=True)
    color_feat = np.concatenate([hist_a, hist_b])

    # --- LBP纹理特征：10维 ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    radius = 2
    n_points = radius * 8
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    hist_lbp, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)

    # 合并：128 + 10 = 138维
    return np.concatenate([color_feat, hist_lbp])

# ========== 3. 批量提取所有图片特征 ==========
X, y = [], []
for label, cls_name in enumerate(classes):
    folder = os.path.join(src_path, cls_name)
    for name in os.listdir(folder):
        if name.endswith((".jpg", ".png")):
            feat = extract_features(os.path.join(folder, name))
            X.append(feat)
            y.append(label)

X = np.array(X)
y = np.array(y)
print("特征矩阵形状：", X.shape)  # (500, 138)

# ========== 4. 划分训练集:测试集=8:2 ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2025, stratify=y
)

# ========== 5. 训练3种分类器并评估 ==========
models = {
    "随机森林": RandomForestClassifier(n_estimators=100, random_state=2025),
    "SVM": SVC(C=100, gamma="scale", random_state=2025),
    "逻辑回归": LogisticRegression(max_iter=1000)
}

for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test
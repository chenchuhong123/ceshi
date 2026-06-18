import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# 1. 读取数据
df = pd.read_csv("cities.csv")
# 提取特征与城市名称
X = df.iloc[:, 1:9].values  # 消费特征列
city_name = df.iloc[:, 0].values  # 城市名

# 2. 遍历簇数 k=3~6，计算聚类评价指标
k_list = [3, 4, 5, 6]
sil_score = []
ch_score = []
db_score = []

for k in k_list:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    # 计算三大评价指标
    s = silhouette_score(X, labels)
    c = calinski_harabasz_score(X, labels)
    d = davies_bouldin_score(X, labels)
    sil_score.append(s)
    ch_score.append(c)
    db_score.append(d)
    print(f"聚类数k={k}：")
    print(f"轮廓系数：{s:.4f}，CH指数：{c:.2f}，DB指数：{d:.4f}\n")

# 3. 选择最优k，重新聚类并划分城市
# 根据指标综合选择最优簇数（示例：假设最优k=4）
best_k = 4
kmeans_best = KMeans(n_clusters=best_k, random_state=42)
df["簇标签"] = kmeans_best.fit_predict(X)

# 按簇分组输出城市
print("===== 各聚类簇对应的城市 =====")
for label in range(best_k):
    city_group = df[df["簇标签"] == label]["city"].tolist()
    print(f"第{label+1}簇城市：{city_group}")

# 4. 选做：pairplot可视化（前4个特征+簇标签）
sns.pairplot(df, vars=df.columns[1:5], hue="簇标签", palette="husl")
plt.show()
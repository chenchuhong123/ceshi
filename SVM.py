import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

# 1. 读取数据
data = pd.read_csv("titanic_data.csv")
print("存活/死亡统计：\n", data['Survived'].value_counts())

# 2. 画直方图
sns.displot(data=data, x='SibSp')
sns.displot(data=data, x='Parch')
plt.show()

# 3. 筛选特征、删缺失值
data = data[['Survived','Pclass','Sex','Age','Fare','Embarked']].dropna()

# 4. 编码
data['Sex'] = data['Sex'].map({'male':1,'female':0})
pclass_dum = pd.get_dummies(data['Pclass'], prefix='Pclass')
embarked_dum = pd.get_dummies(data['Embarked'], prefix='Embarked')
X = pd.concat([data[['Sex','Age','Fare']], pclass_dum, embarked_dum], axis=1)
y = data['Survived']

# 5. 划分7:3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2025)

# 6. 标准化Age、Fare
X_train = X_train.values
X_test = X_test.values
age_scaler = MinMaxScaler()
X_train[:,1] = age_scaler.fit_transform(X_train[:,1:2])
X_test[:,1] = age_scaler.transform(X_test[:,1:2])
fare_scaler = MinMaxScaler()
X_train[:,2] = fare_scaler.fit_transform(X_train[:,2:3])
X_test[:,2] = fare_scaler.transform(X_test[:,2:3])

# 7. SVM基础模型
svm = SVC(C=100, random_state=2025)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print("混淆矩阵：\n", confusion_matrix(y_test, y_pred))
print("分类报告：\n", classification_report(y_test, y_pred))

# 8. 调参
params = {'C': [1,10,50,100,200,500,1000]}
gs = GridSearchCV(SVC(random_state=2025), params, cv=5)
gs.fit(X_train, y_train)
print("最优C：", gs.best_params_)
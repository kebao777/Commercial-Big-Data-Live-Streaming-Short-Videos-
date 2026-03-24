# =========================
# 第二部分：城市级别、婚姻状况与用户流失分析
# =========================
import warnings

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False



# 读取数据
excel_file = pd.ExcelFile('直播电商数据集.xlsx')
# 数据预览
df = excel_file.parse('E Comm')
print(f'sheet表名为{'E Comm'}的基本信息：')
df.info()

#填充Tenure列的缺失值
df['Tenure'].fillna(df['Tenure'].median(), inplace=True)
#填充WarehouseToHome列的缺失值
df['WarehouseToHome'].fillna(df['WarehouseToHome'].median(), inplace=True)
#填充HourSpendOnApp列的缺失值
df['HourSpendOnApp'].fillna(df['HourSpendOnApp'].mean(), inplace=True)
#填充OrderCount列的缺失值
df['OrderCount'].fillna(0, inplace=True)
#填充OrderAmountHikeFromlastYear列的缺失值
df['OrderAmountHikeFromlastYear'].fillna(df['OrderAmountHikeFromlastYear'].median(), inplace=True)
#填充CouponUsed列的缺失值
df['CouponUsed'].fillna(df['CouponUsed'].median(), inplace=True)
#填充DaySinceLastOrder列的缺失值
df['DaySinceLastOrder'].fillna(df['DaySinceLastOrder'].median(), inplace=True)


# =========================
# 1. 描述性统计分析
# =========================

print("\n========== 一、描述性统计分析 ==========")

# 用户流失总人数
churn_total = (df['Churn'] == 1).sum()
total_users = len(df)
churn_rate = churn_total / total_users

print("用户总人数：", total_users)
print("流失用户总人数：", churn_total)
print("整体流失率：{:.2%}".format(churn_rate))

# 各城市级别下的流失人数、总人数、流失率
citytier_summary = (
    df.groupby('CityTier')
      .agg(
          总人数=('Churn', 'count'),
          流失人数=('Churn', lambda x: (x == 1).sum())
      )
      .reset_index()
)
citytier_summary['流失率'] = citytier_summary['流失人数'] / citytier_summary['总人数']

print("\n各城市级别流失情况：")
print(citytier_summary)

# 各婚姻状况下的流失人数、总人数、流失率
marital_summary = (
    df.groupby('MaritalStatus')
      .agg(
          总人数=('Churn', 'count'),
          流失人数=('Churn', lambda x: (x == 1).sum())
      )
      .reset_index()
)
marital_summary['流失率'] = marital_summary['流失人数'] / marital_summary['总人数']

print("\n各婚姻状况流失情况：")
print(marital_summary)

# =========================
# 2. 可视化分析
# =========================

print("\n========== 二、可视化分析 ==========")

# -------------------------
# 2.1 城市级别下的流失人数柱状图
# -------------------------
plt.figure(figsize=(8, 5))
plt.bar(citytier_summary['CityTier'].astype(str), citytier_summary['流失人数'], edgecolor='gray')
plt.xlabel('城市级别')
plt.ylabel('流失人数')
plt.title('不同城市级别下的用户流失人数')

for i, v in enumerate(citytier_summary['流失人数']):
    plt.text(i, v, str(v), ha='center', va='bottom')

plt.show()

# -------------------------
# 2.2 婚姻状况下的流失人数柱状图
# -------------------------
plt.figure(figsize=(8, 5))
plt.bar(marital_summary['MaritalStatus'].astype(str), marital_summary['流失人数'], edgecolor='gray')
plt.xlabel('婚姻状况')
plt.ylabel('流失人数')
plt.title('不同婚姻状况下的用户流失人数')

for i, v in enumerate(marital_summary['流失人数']):
    plt.text(i, v, str(v), ha='center', va='bottom')

plt.show()

# -------------------------
# 2.3 城市级别流失用户分布饼图
# -------------------------
citytier_churn_only = df[df['Churn'] == 1].groupby('CityTier')['CustomerID'].count()

plt.figure(figsize=(6, 6))
plt.pie(
    citytier_churn_only.values,
    labels=citytier_churn_only.index.astype(str),
    autopct='%.2f%%',
    pctdistance=1.15,
    labeldistance=1.05
)
plt.title('流失用户在不同城市级别中的分布')
plt.show()

# -------------------------
# 2.4 婚姻状况流失用户分布饼图
# -------------------------
marital_churn_only = df[df['Churn'] == 1].groupby('MaritalStatus')['CustomerID'].count()

plt.figure(figsize=(6, 6))
plt.pie(
    marital_churn_only.values,
    labels=marital_churn_only.index.astype(str),
    autopct='%.2f%%',
    pctdistance=1.15,
    labeldistance=1.05
)
plt.title('流失用户在不同婚姻状况中的分布')
plt.show()

# =========================
# 2.7 boxplot：不同城市级别下“各婚姻状况流失率”的分布
# =========================

# 先按 城市级别 + 婚姻状况 分组，计算每个交叉组的流失率
cross_rate = (
    df.groupby(['CityTier', 'MaritalStatus'])['Churn']
      .mean()
      .reset_index(name='ChurnRate')
)

# 按城市级别整理成箱线图输入数据
citytier_box_data = []
citytier_labels = sorted(cross_rate['CityTier'].unique())

for city in citytier_labels:
    rates = cross_rate.loc[cross_rate['CityTier'] == city, 'ChurnRate'].values
    citytier_box_data.append(rates)

plt.figure(figsize=(8, 5))
bp = plt.boxplot(citytier_box_data, tick_labels=citytier_labels)
plt.title('不同城市级别下流失率分布（按婚姻状况细分）')
plt.xlabel('城市级别')
plt.ylabel('流失率')


# 标注每个组内的具体数值
for i, rates in enumerate(citytier_box_data, start=1):
    for j, y in enumerate(rates):
        plt.text(i + 0.08, y, f'{y:.3f}', fontsize=9, color='blue', va='center')

plt.show()


# =========================
# 2.8 boxplot：不同婚姻状况下“各城市级别流失率”的分布
# =========================

marital_labels = list(cross_rate['MaritalStatus'].unique())
marital_box_data = []

for status in marital_labels:
    rates = cross_rate.loc[cross_rate['MaritalStatus'] == status, 'ChurnRate'].values
    marital_box_data.append(rates)

plt.figure(figsize=(8, 5))
bp = plt.boxplot(marital_box_data, tick_labels=marital_labels)
plt.title('不同婚姻状况下流失率分布（按城市级别细分）')
plt.xlabel('婚姻状况')
plt.ylabel('流失率')

# 标注每个组内的具体数值
for i, rates in enumerate(marital_box_data, start=1):
    for j, y in enumerate(rates):
        plt.text(i + 0.08, y, f'{y:.3f}', fontsize=9, color='blue', va='center')

plt.show()

# =========================
# 不同婚姻状况下流失用户与未流失用户占比（堆叠柱状图）
# =========================

# 获取婚姻状况标签，按字母顺序排序
marital_labels = sorted(df['MaritalStatus'].dropna().unique())

# 统计各婚姻状况下流失与未流失人数
y1 = [list(df.loc[df['MaritalStatus'] == status, 'Churn']).count(1) for status in marital_labels]  # 流失
y2 = [list(df.loc[df['MaritalStatus'] == status, 'Churn']).count(0) for status in marital_labels]  # 未流失

data = np.array([y1, y2])
x = np.arange(len(marital_labels))
bottom_y = np.zeros(len(marital_labels))
sums = np.sum(data, axis=0)

colors = ['#fc8d62', '#66c2a5']
legend_labels = ['流失用户占比', '未流失用户占比']

figure, ax = plt.subplots(figsize=(10, 6))

for j, i in enumerate(data):
    y = i / sums
    ax.bar(
        x,
        y,
        width=0.6,
        color=colors[j],
        bottom=bottom_y,
        edgecolor='gray'
    )
    bottom_y += y

ax.set_xticks(x)
ax.set_xticklabels(marital_labels)
ax.yaxis.set_major_formatter(PercentFormatter(1))

patches = [
    mpatches.Patch(color=colors[h], label=legend_labels[h])
    for h in range(len(legend_labels))
]
ax.legend(handles=patches, ncol=1, bbox_to_anchor=(1, 1))
figure.subplots_adjust(right=0.8)

# 计算各婚姻状况下流失率和未流失率
Y_churn1 = []
Y_churn0 = []

for status in marital_labels:
    df_status = df.loc[df['MaritalStatus'] == status]
    Y_churn1.append(list(df_status['Churn']).count(1) / len(df_status))
    Y_churn0.append(list(df_status['Churn']).count(0) / len(df_status))

# 在柱子中间显示百分比
for a, b1, b0 in zip(x, Y_churn1, Y_churn0):
    ax.text(a, b1 / 2, '%.2f%%' % (b1 * 100), ha='center', va='center')
    ax.text(a, b1 + b0 / 2, '%.2f%%' % (b0 * 100), ha='center', va='center')

ax.set_ylabel('流失用户与未流失用户占比', fontsize=13)
ax.set_xlabel('婚姻状况', fontsize=13)
ax.set_title('不同婚姻状况下用户流失情况')
plt.show()


# =========================
# 不同城市级别下流失用户与未流失用户占比（堆叠柱状图）
# =========================

# 获取城市级别标签，按数值升序排序
city_labels = sorted(df['CityTier'].dropna().unique())

# 统计各城市级别下流失与未流失人数
y1 = [list(df.loc[df['CityTier'] == city, 'Churn']).count(1) for city in city_labels]  # 流失
y2 = [list(df.loc[df['CityTier'] == city, 'Churn']).count(0) for city in city_labels]  # 未流失

data = np.array([y1, y2])
x = np.arange(len(city_labels))
bottom_y = np.zeros(len(city_labels))
sums = np.sum(data, axis=0)

colors = ['#8da0cb', '#a6d854']
legend_labels = ['流失用户占比', '未流失用户占比']

figure, ax = plt.subplots(figsize=(10, 6))

for j, i in enumerate(data):
    y = i / sums
    ax.bar(
        x,
        y,
        width=0.6,
        color=colors[j],
        bottom=bottom_y,
        edgecolor='gray'
    )
    bottom_y += y

ax.set_xticks(x)
ax.set_xticklabels(city_labels)
ax.yaxis.set_major_formatter(PercentFormatter(1))

patches = [
    mpatches.Patch(color=colors[h], label=legend_labels[h])
    for h in range(len(legend_labels))
]
ax.legend(handles=patches, ncol=1, bbox_to_anchor=(1, 1))
figure.subplots_adjust(right=0.8)

# 计算各城市级别下流失率和未流失率
Y_churn1 = []
Y_churn0 = []

for city in city_labels:
    df_city = df.loc[df['CityTier'] == city]
    Y_churn1.append(list(df_city['Churn']).count(1) / len(df_city))
    Y_churn0.append(list(df_city['Churn']).count(0) / len(df_city))

# 在柱子中间显示百分比
for a, b1, b0 in zip(x, Y_churn1, Y_churn0):
    ax.text(a, b1 / 2, '%.2f%%' % (b1 * 100), ha='center', va='center')
    ax.text(a, b1 + b0 / 2, '%.2f%%' % (b0 * 100), ha='center', va='center')

ax.set_ylabel('流失用户与未流失用户占比', fontsize=13)
ax.set_xlabel('城市级别', fontsize=13)
ax.set_title('不同城市级别下用户流失情况')
plt.show()
# =========================
# 3. 关联分析：逻辑回归
# =========================

print("\n========== 三、关联分析：逻辑回归 ==========")

# 只选取城市级别、婚姻状况作为特征
X = df[['CityTier', 'MaritalStatus']].copy()
y = df['Churn']

# 对婚姻状况做独热编码
X = pd.get_dummies(X, columns=['MaritalStatus'], drop_first=True)

print("\n建模特征：")
print(X.head())

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 建立逻辑回归模型
lr_model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced'
)

# 训练模型
lr_model.fit(X_train, y_train)

# 预测
y_pred = lr_model.predict(X_test)

# 模型评估
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n逻辑回归模型评价指标：")
print("Accuracy（准确率）: {:.4f}".format(acc))
print("Precision（精确率）: {:.4f}".format(precision))
print("Recall（召回率）: {:.4f}".format(recall))
print("F1-score: {:.4f}".format(f1))

print("\n分类报告：")
print(classification_report(y_test, y_pred, digits=4, zero_division=0))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print("\n混淆矩阵：")
print(cm)

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['不流失', '流失'])
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.title('城市级别、婚姻状况对用户流失影响的逻辑回归混淆矩阵')
plt.show()

# 回归系数
coef_df = pd.DataFrame({
    '特征': X.columns,
    '回归系数': lr_model.coef_[0]
}).sort_values(by='回归系数', ascending=False)

print("\n逻辑回归回归系数：")
print(coef_df)

# 可视化回归系数
plt.figure(figsize=(10, 5))
plt.bar(coef_df['特征'], coef_df['回归系数'], edgecolor='gray')
plt.axhline(0, color='black', linewidth=1)
plt.xlabel('特征')
plt.ylabel('回归系数')
plt.title('城市级别、婚姻状况对用户流失影响的逻辑回归系数')

for i, v in enumerate(coef_df['回归系数']):
    plt.text(i, v, f'{v:.3f}', ha='center', va='bottom' if v >= 0 else 'top')

plt.xticks(rotation=20)
plt.show()

# =========================
# 4. 输出结论辅助写PPT
# =========================

print("\n========== 四、结论辅助输出 ==========")

max_city = citytier_summary.sort_values(by='流失率', ascending=False).iloc[0]
max_marital = marital_summary.sort_values(by='流失率', ascending=False).iloc[0]

print("流失率最高的城市级别：CityTier = {}，流失率 = {:.2%}".format(
    max_city['CityTier'], max_city['流失率'])
)

print("流失率最高的婚姻状况：{}，流失率 = {:.2%}".format(
    max_marital['MaritalStatus'], max_marital['流失率'])
)
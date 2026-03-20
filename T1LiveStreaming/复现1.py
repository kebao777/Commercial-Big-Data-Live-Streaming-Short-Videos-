import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import warnings
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter

warnings.filterwarnings("ignore", category=FutureWarning)

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


#分析用户登录设备（饼图）
df_churn1 = df.loc[df['Churn'] == 1]  #获取流失的用户
#按照登录设备分组后计算数据个数
df_churn1_PreferredLoginDevice = df_churn1.groupby(['PreferredLoginDevice'])['CustomerID'].count().reset_index().rename(columns={'CustomerID': 'count'})
label_churn1_PreferredLoginDevice=df_churn1_PreferredLoginDevice['PreferredLoginDevice']  #提取标签
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.pie(df_churn1_PreferredLoginDevice['count'],
        #传入标签
 labels=label_churn1_PreferredLoginDevice.values,
        #格式化输出百分比
        autopct='%.2f%%',
pctdistance=1.2,labeldistance=1.05
 )
plt.show()
 #分析未流失用户的首选登录设备（饼图）
df_churn0 = df.loc[df['Churn'] == 0]  #获取未流失的用户
#按照登录设备分组后计算数据个数
df_churn0_PreferredLoginDevice = df_churn0.groupby(['PreferredLoginDevice'])['CustomerID'].count().reset_index().rename(columns={'CustomerID': 'count'})
label_churn0_PreferredLoginDevice=df_churn0_PreferredLoginDevice['PreferredLoginDevice']  #提取标签
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号
plt.pie(df_churn0_PreferredLoginDevice['count'],
        #传入标签
        labels=label_churn1_PreferredLoginDevice.values,
        #格式化输出百分比
        autopct='%.2f%%',
 pctdistance=1.24,
 labeldistance=1.05,
 )
plt.show()


# 分析用户在首选登录设备不同情况下的流失情况（堆积柱状图）

# 提取不同首选登录设备的用户
df_MobilePhone = df.loc[df['PreferredLoginDevice'] == 'Mobile Phone']   # 首选移动手机
df_Phone = df.loc[df['PreferredLoginDevice'] == 'Phone']                # 首选普通电话
df_Pad = df.loc[df['PreferredLoginDevice'] == 'Pad']                    # 首选平板计算机

# 统计流失与未流失人数
y1 = [
    list(df_MobilePhone['Churn']).count(1),
    list(df_Phone['Churn']).count(1),
    list(df_Pad['Churn']).count(1)
]  # 流失

y2 = [
    list(df_MobilePhone['Churn']).count(0),
    list(df_Phone['Churn']).count(0),
    list(df_Pad['Churn']).count(0)
]  # 未流失

# 为方便后续引用，将数据放入 data
data = [y1, y2]

# 提取标签
label_churn0_PreferredLoginDevice = ['MobilePhone', 'Phone', 'Pad']

# 获取 x 轴标签位置
x = range(len(label_churn0_PreferredLoginDevice))

# 将 bottom_y 元素都初始化为 0
bottom_y = np.zeros(len(label_churn0_PreferredLoginDevice))

# 将数据放入数组中
data = np.array(data)

# 求数组 data 的和，为计算百分比做准备
sums = np.sum(data, axis=0)

j = 0
colors = ['#66c2a5', '#8da0cb']

# 创建子图
figure, ax = plt.subplots()

# 绘制堆积柱状图
for i in data:
    y = i / sums   # 获取各个 y 值的百分比
    plt.bar(
        x,
        y,
        width=0.5,
        color=np.array(colors)[j],
        bottom=bottom_y,
        edgecolor='gray'
    )
    bottom_y = y + bottom_y   # 实现百分比柱子的堆积
    plt.xticks(x, label_churn0_PreferredLoginDevice)  # 设置 x 轴坐标标签
    j += 1

# 设置图例标签
legend_labels = ['流失用户占比', '未流失用户占比']
color = ['#66c2a5', '#8da0cb']

# 将颜色和图例标签对应
patches = [
    mpatches.Patch(color=color[h], label='{:s}'.format(legend_labels[h]))
    for h in range(len(legend_labels))
]

ax = plt.gca()   # 绘制子图
box = ax.get_position()

# y 轴设置为百分比
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

# 显示图例，并设置图例位置
ax.legend(handles=patches, ncol=1, bbox_to_anchor=(1, 1))
figure.subplots_adjust(right=0.7)

# 计算不同首选登录设备下未流失、流失用户占比
Y_churn0 = [
    list(df_MobilePhone['Churn']).count(0) / len(df_MobilePhone),
    list(df_Phone['Churn']).count(0) / len(df_Phone),
    list(df_Pad['Churn']).count(0) / len(df_Pad)
]  # 未流失用户占比

Y_churn1 = [
    list(df_MobilePhone['Churn']).count(1) / len(df_MobilePhone),
    list(df_Phone['Churn']).count(1) / len(df_Phone),
    list(df_Pad['Churn']).count(1) / len(df_Pad)
]  # 流失用户占比

# 柱子上的数字显示
for a, b in zip(x, Y_churn0):
    plt.text(a, b, '%.2f%%' % (b * 100), ha='center', va='bottom')

for a, b in zip(x, Y_churn1):
    plt.text(a, b, '%.2f%%' % (b * 100), ha='center', va='bottom')

# 设置坐标轴标签
ax.set_ylabel('流失用户与未流失用户占比', fontsize=13)
ax.set_xlabel('用户首选登录设备', fontsize=13)

plt.show()

# =========================
# 10. 用户属性特征分析——性别分析（饼图）
# =========================

# 分析流失用户的性别分布
df_churn1 = df.loc[df['Churn'] == 1]
df_churn1_Gender = (
    df_churn1.groupby('Gender')['CustomerID']
    .count()
    .reset_index()
    .rename(columns={'CustomerID': 'count'})
)

label_churn1_Gender = df_churn1_Gender['Gender']

plt.figure(figsize=(6, 6))
plt.pie(
    df_churn1_Gender['count'],
    labels=label_churn1_Gender.values,
    autopct='%.2f%%',
    pctdistance=1.2,
    labeldistance=1.05
)
plt.title('流失用户性别分布')
plt.show()


# 分析未流失用户的性别分布
df_churn0 = df.loc[df['Churn'] == 0]
df_churn0_Gender = (
    df_churn0.groupby('Gender')['CustomerID']
    .count()
    .reset_index()
    .rename(columns={'CustomerID': 'count'})
)

label_churn0_Gender = df_churn0_Gender['Gender']

plt.figure(figsize=(6, 6))
plt.pie(
    df_churn0_Gender['count'],
    labels=label_churn0_Gender.values,
    autopct='%.2f%%',
    pctdistance=1.2,
    labeldistance=1.05
)
plt.title('未流失用户性别分布')
plt.show()

# =========================
# 11. 用户属性特征分析——性别分析（堆积柱状图）
# =========================

# 提取不同性别用户
df_Female = df.loc[df['Gender'] == 'Female']
df_Male = df.loc[df['Gender'] == 'Male']

# 统计流失与未流失人数
y1 = [
    list(df_Female['Churn']).count(1),
    list(df_Male['Churn']).count(1)
]  # 流失

y2 = [
    list(df_Female['Churn']).count(0),
    list(df_Male['Churn']).count(0)
]  # 未流失

data = np.array([y1, y2])

labels_gender = ['Female', 'Male']
x = np.arange(len(labels_gender))
bottom_y = np.zeros(len(labels_gender))
sums = np.sum(data, axis=0)

#colors = ['#fc8d62', '#66c2a5']
legend_labels = ['流失用户占比', '未流失用户占比']

figure, ax = plt.subplots(figsize=(8, 5))

for j, i in enumerate(data):
    y = i / sums
    ax.bar(
        x,
        y,
        width=0.5,
        color=colors[j],
        bottom=bottom_y,
        edgecolor='gray'
    )
    bottom_y += y

ax.set_xticks(x)
ax.set_xticklabels(labels_gender)
ax.yaxis.set_major_formatter(PercentFormatter(1))

patches = [
    mpatches.Patch(color=colors[h], label=legend_labels[h])
    for h in range(len(legend_labels))
]
ax.legend(handles=patches, ncol=1, bbox_to_anchor=(1, 1))
figure.subplots_adjust(right=0.78)

# 计算不同性别流失/未流失占比
Y_churn0 = [
    list(df_Female['Churn']).count(0) / len(df_Female),
    list(df_Male['Churn']).count(0) / len(df_Male)
]

Y_churn1 = [
    list(df_Female['Churn']).count(1) / len(df_Female),
    list(df_Male['Churn']).count(1) / len(df_Male)
]

# 在柱子中间显示百分比
for a, b1, b0 in zip(x, Y_churn1, Y_churn0):
    ax.text(a, b1 / 2, '%.2f%%' % (b1 * 100), ha='center', va='center')
    ax.text(a, b1 + b0 / 2, '%.2f%%' % (b0 * 100), ha='center', va='center')

ax.set_ylabel('流失用户与未流失用户占比', fontsize=13)
ax.set_xlabel('用户性别', fontsize=13)
ax.set_title('不同性别用户流失情况')
plt.show()

# =========================
# 12. 用户属性特征分析——年龄分析（饼图）
# =========================

# 年龄分组映射
age_map = {
    1: '10-19',
    2: '20-29',
    3: '30-39',
    4: '40-49',
    5: '50-59',
    6: '60-69'
}

# 流失用户年龄分布
df_churn1_AgeGroup = (
    df_churn1.groupby('AgeGroup')['CustomerID']
    .count()
    .reset_index()
    .rename(columns={'CustomerID': 'count'})
)

df_churn1_AgeGroup['AgeLabel'] = df_churn1_AgeGroup['AgeGroup'].map(age_map)

plt.figure(figsize=(7, 7))
plt.pie(
    df_churn1_AgeGroup['count'],
    labels=df_churn1_AgeGroup['AgeLabel'].values,
    autopct='%.2f%%',
    pctdistance=1.2,
    labeldistance=1.05
)
plt.title('流失用户年龄分布')
plt.show()

# 未流失用户年龄分布
df_churn0_AgeGroup = (
    df_churn0.groupby('AgeGroup')['CustomerID']
    .count()
    .reset_index()
    .rename(columns={'CustomerID': 'count'})
)

df_churn0_AgeGroup['AgeLabel'] = df_churn0_AgeGroup['AgeGroup'].map(age_map)

plt.figure(figsize=(7, 7))
plt.pie(
    df_churn0_AgeGroup['count'],
    labels=df_churn0_AgeGroup['AgeLabel'].values,
    autopct='%.2f%%',
    pctdistance=1.2,
    labeldistance=1.05
)
plt.title('未流失用户年龄分布')
plt.show()

# =========================
# 13. 用户属性特征分析——年龄分析（堆积柱状图）
# =========================

age_order = [1, 2, 3, 4, 5, 6]
age_labels = [age_map[i] for i in age_order]

# 各年龄段流失、未流失人数
y1 = [list(df.loc[df['AgeGroup'] == i, 'Churn']).count(1) for i in age_order]  # 流失
y2 = [list(df.loc[df['AgeGroup'] == i, 'Churn']).count(0) for i in age_order]  # 未流失

data = np.array([y1, y2])
x = np.arange(len(age_labels))
bottom_y = np.zeros(len(age_labels))
sums = np.sum(data, axis=0)

#colors = ['#e78ac3', '#8da0cb']
legend_labels = ['流失用户占比', '未流失用户占比']

figure, ax = plt.subplots(figsize=(10, 5))

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
ax.set_xticklabels(age_labels)
ax.yaxis.set_major_formatter(PercentFormatter(1))

patches = [
    mpatches.Patch(color=colors[h], label=legend_labels[h])
    for h in range(len(legend_labels))
]
ax.legend(handles=patches, ncol=1, bbox_to_anchor=(1, 1))
figure.subplots_adjust(right=0.8)

# 计算各年龄组流失率/未流失率
Y_churn1 = []
Y_churn0 = []

for i in age_order:
    df_age = df.loc[df['AgeGroup'] == i]
    Y_churn1.append(list(df_age['Churn']).count(1) / len(df_age))
    Y_churn0.append(list(df_age['Churn']).count(0) / len(df_age))

# 柱内显示百分比
for a, b1, b0 in zip(x, Y_churn1, Y_churn0):
    ax.text(a, b1 / 2, '%.1f%%' % (b1 * 100), ha='center', va='center', fontsize=9)
    ax.text(a, b1 + b0 / 2, '%.1f%%' % (b0 * 100), ha='center', va='center', fontsize=9)

ax.set_ylabel('流失用户与未流失用户占比', fontsize=13)
ax.set_xlabel('用户年龄段', fontsize=13)
ax.set_title('不同年龄段用户流失情况')
plt.show()

# =========================
# 14. 输出各年龄组流失率，便于写PPT结论
# =========================

age_churn_summary = pd.DataFrame({
    '年龄段': age_labels,
    '流失人数': y1,
    '未流失人数': y2,
    '流失率': [round(v * 100, 2) for v in Y_churn1]
})

print('各年龄段流失情况：')
print(age_churn_summary)

# =========================
# 15. 用户行为特征分析——计算整体流失率
# =========================

churn_count = (df['Churn'] == 1).sum()
total_count = len(df)
churn_rate = churn_count / total_count

print('流失用户数量：', churn_count)
print('总用户数量：', total_count)
print('用户流失率：{:.2%}'.format(churn_rate))

# =========================
# 16. 用户行为特征分析——最近一个月订单偏好类型分析（饼图）
# =========================

# 流失用户订单偏好
df_churn1_OrderCat = (
    df_churn1.groupby('PreferedOrderCat')['CustomerID']
    .count()
    .reset_index()
    .rename(columns={'CustomerID': 'count'})
    .sort_values(by='count', ascending=False)
)

plt.figure(figsize=(8, 8))
plt.pie(
    df_churn1_OrderCat['count'],
    labels=df_churn1_OrderCat['PreferedOrderCat'].values,
    autopct='%.2f%%',
    pctdistance=1.18,
    labeldistance=1.05
)
plt.title('流失用户最近一个月订单偏好类型分布')
plt.show()

# 未流失用户订单偏好
df_churn0_OrderCat = (
    df_churn0.groupby('PreferedOrderCat')['CustomerID']
    .count()
    .reset_index()
    .rename(columns={'CustomerID': 'count'})
    .sort_values(by='count', ascending=False)
)

plt.figure(figsize=(8, 8))
plt.pie(
    df_churn0_OrderCat['count'],
    labels=df_churn0_OrderCat['PreferedOrderCat'].values,
    autopct='%.2f%%',
    pctdistance=1.18,
    labeldistance=1.05
)
plt.title('未流失用户最近一个月订单偏好类型分布')
plt.show()

# =========================
# 17. 用户行为特征分析——最近一个月订单偏好类型分析（堆积柱状图）
# =========================

order_cat_labels = sorted(df['PreferedOrderCat'].dropna().unique())

# 统计每类订单偏好下流失与未流失人数
y1 = [list(df.loc[df['PreferedOrderCat'] == cat, 'Churn']).count(1) for cat in order_cat_labels]  # 流失
y2 = [list(df.loc[df['PreferedOrderCat'] == cat, 'Churn']).count(0) for cat in order_cat_labels]  # 未流失

data = np.array([y1, y2])
x = np.arange(len(order_cat_labels))
bottom_y = np.zeros(len(order_cat_labels))
sums = np.sum(data, axis=0)

#colors = ['#a6d854', '#ffd92f']
legend_labels = ['流失用户占比', '未流失用户占比']

figure, ax = plt.subplots(figsize=(12, 6))

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
ax.set_xticklabels(order_cat_labels, rotation=30)
ax.yaxis.set_major_formatter(PercentFormatter(1))

patches = [
    mpatches.Patch(color=colors[h], label=legend_labels[h])
    for h in range(len(legend_labels))
]
ax.legend(handles=patches, ncol=1, bbox_to_anchor=(1, 1))
figure.subplots_adjust(right=0.82, bottom=0.2)

# 计算各类别流失率和未流失率
Y_churn1 = []
Y_churn0 = []

for cat in order_cat_labels:
    df_cat = df.loc[df['PreferedOrderCat'] == cat]
    Y_churn1.append(list(df_cat['Churn']).count(1) / len(df_cat))
    Y_churn0.append(list(df_cat['Churn']).count(0) / len(df_cat))

# 显示流失率（只标流失率，避免太拥挤）
for a, b in zip(x, Y_churn1):
    ax.text(a, b / 2, '%.1f%%' % (b * 100), ha='center', va='center', fontsize=8)

ax.set_ylabel('流失用户与未流失用户占比', fontsize=13)
ax.set_xlabel('最近一个月订单偏好类型', fontsize=13)
ax.set_title('不同订单偏好类型下用户流失情况')
plt.show()

# =========================
# 18. 输出各订单偏好类别流失率，便于写PPT结论
# =========================

order_cat_summary = pd.DataFrame({
    '订单偏好类型': order_cat_labels,
    '流失人数': y1,
    '未流失人数': y2,
    '流失率': [round(v * 100, 2) for v in Y_churn1]
}).sort_values(by='流失率', ascending=False)

print('各订单偏好类型流失情况：')
print(order_cat_summary)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


feature_cols = ['HourSpendOnApp', 'Complain', 'DiscountAmount', 'DaySinceLastOrder']

target_col = 'Churn'

print("\n用于建模的特征：", feature_cols)

# 提取特征和标签
X = df[feature_cols]
y = df[target_col]

print("\n特征数据预览：")
print(X.head())

print("\n标签分布：")
print(y.value_counts())

# =========================
# 逻辑回归
# =========================

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 第24页：数据准备、训练集测试集划分、模型训练
# =========================

# 拆分训练集和测试集
# test_size=0.2 表示 80%训练，20%测试
# stratify=y 保持正负样本比例一致
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n训练集形状：", X_train.shape)
print("测试集形状：", X_test.shape)

# 构建逻辑回归模型
# class_weight='balanced' 用于缓解类别不平衡问题
lr_model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced'
)

# 模型训练
lr_model.fit(X_train, y_train)

# 预测
y_pred = lr_model.predict(X_test)
y_prob = lr_model.predict_proba(X_test)[:, 1]

# =========================
# 结果评估与模型表现
# =========================

# 计算评价指标
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n========== 逻辑回归模型评价指标 ==========")
print("Accuracy（准确率）: {:.4f}".format(acc))
print("Precision（精确率）: {:.4f}".format(precision))
print("Recall（召回率）: {:.4f}".format(recall))
print("F1-score: {:.4f}".format(f1))

print("\n分类报告：")
print(classification_report(y_test, y_pred, digits=4, zero_division=0))

# =========================
# 混淆矩阵
# =========================

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\n========== 混淆矩阵 ==========")
print(cm)
print("True Negative (TN):", tn)
print("False Positive (FP):", fp)
print("False Negative (FN):", fn)
print("True Positive (TP):", tp)

# 绘制混淆矩阵
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['不流失(0)', '流失(1)'])
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.title('逻辑回归模型 - 混淆矩阵')
plt.show()

# =========================
# 模型回归系数
# =========================

coef_df = pd.DataFrame({
    '特征': feature_cols,
    '回归系数': lr_model.coef_[0]
}).sort_values(by='回归系数', ascending=False)

print("\n========== 模型回归系数 ==========")
print(coef_df)

# 绘制回归系数图
plt.figure(figsize=(8, 5))
bars = plt.bar(coef_df['特征'], coef_df['回归系数'], edgecolor='gray')
plt.axhline(0, color='black', linewidth=1)
plt.title('逻辑回归模型回归系数')
plt.xlabel('特征')
plt.ylabel('系数')

for i, v in enumerate(coef_df['回归系数']):
    plt.text(i, v, f'{v:.3f}', ha='center', va='bottom' if v >= 0 else 'top')

plt.show()

# =========================
# 输出每个用户的流失概率
# =========================

pred_result = pd.DataFrame({
    '真实值': y_test.values,
    '预测值': y_pred,
    '预测为流失的概率': y_prob
})

print("\n预测结果预览：")
print(pred_result.head(10))

# 按流失概率从高到低查看前10个高风险用户
pred_result_sorted = pred_result.sort_values(by='预测为流失的概率', ascending=False)
print("\n预测流失概率最高的前10个样本：")
print(pred_result_sorted.head(10))
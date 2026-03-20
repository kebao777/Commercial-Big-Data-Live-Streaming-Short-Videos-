import re
import warnings

import jieba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from wordcloud import WordCloud

warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# =========================
# 1. 读取数据与基础预览
# =========================
file_path = "短视频营销数据.xlsx"
sheet_name = 0                      # 如果有指定 sheet，可改成具体名称

df = pd.read_excel(file_path, sheet_name=sheet_name)

print("数据前5行：")
print(df.head())

print("\n数据基本信息：")
print(df.info())

print("\n缺失值统计：")
print(df.isnull().sum())


# =========================
# 2. 缺失值处理
# video_title 存在缺失，可基于 product_title 补
# =========================
if "video_title" in df.columns and "product_title" in df.columns:
    df["video_title"] = df["video_title"].fillna("")

    # 如果 video_title 为空，就用 product_title 的简化文本代替
    empty_mask = df["video_title"].str.strip() == ""
    df.loc[empty_mask, "video_title"] = df.loc[empty_mask, "product_title"].astype(str)

print("\n处理后的缺失值统计：")
print(df.isnull().sum())


# =========================
# 3. 数值变量描述性统计
# =========================
numeric_cols = df.select_dtypes(include=[np.number]).columns
desc = df[numeric_cols].describe().round(2)

print("\n数值型变量 describe() 结果：")
print(desc)

# 以 sale_count 为例计算众数、极差、方差
if "sale_count" in df.columns:
    sale_mode = df["sale_count"].mode()
    sale_range = np.ptp(df["sale_count"].dropna())
    sale_var = np.var(df["sale_count"].dropna())

    print("\nsale_count 统计：")
    print("众数：", sale_mode.iloc[0] if len(sale_mode) > 0 else None)
    print("极差：", sale_range)
    print("方差：", sale_var)


# =========================
# 4. 商品价格分析
# =========================
#商品价格分析
#直方图
plt.figure(figsize=(16,9))  #图像大小
plt.hist(df['price'], bins=9)  #直方图
plt.xlabel('价格')  #横轴标签
plt.ylabel('频数')  #纵轴标签
plt.show()

#饼图
#价格分类规则
def price_group(price):
    if price <= 300:
        return '低价商品'
    elif 300 < price <= 1000:
        return '中价商品'
    else:
        return '高价商品'

df['price_group'] = df['price'].apply(price_group)  #对商品价格进行等级分类
price_group_count = df['price_group'].value_counts(normalize=True)  #统计每个类别占比
plt.pie(price_group_count,
        labels=price_group_count.index,  #不同类别对应的价格分类标签
        autopct='%.1f%%',  #保留一位小数
        labeldistance=1.245,  #标签标注位置与数据块径向距离的比率为1.24
        pctdistance=1.125)  #数据标注位置与数据块径向距离的比率为1.125
plt.show()

# =========================
# 5. 商品关键词分析（词云）
# 这里用 product_title
# =========================
# 商品关键词分析

# Matplotlib 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

content = ""
for i in range(len(df)):   # 遍历每行数据
    content += df["product_title"][i]   # 提取标题内容

w1 = jieba.cut(content, cut_all=True)   # 生成分词列表
w1_space_split = " ".join(w1)           # 用空格连接成字符串文本

wc = WordCloud(
    font_path="C:\\Windows\\Fonts\\msyh.ttc",   # 词云字体设置
    background_color="white",                   # 设置背景颜色为白色
    width=1000,
    height=600
).generate(w1_space_split)                      # 加载词云文本

wc.to_file("词云.png")
# =========================
# 第11页：商品品类分析
# =========================
#商品品类分析
#分词
subtitle = []
for each in df['product_title']:  #遍历每行数据
    k = list(jieba.cut(each))  #对标题列进行分词，组成列表
    subtitle.append(k)  #将列表作为子标题内容
df['subtitle'] = subtitle  #将子标题列添加到数据集
print(df[['product_title', 'subtitle']].head(0))  #字段对比

#第一列是商品所属品类，后续是相关关键词
basic_data = """美妆 化妆 彩妆 美容 眉笔 腮红 眼影 粉底液 修容 遮瑕 口红 睫毛 粉扑 定妆 面霜 颈霜 素颜霜 卸妆
个护 洗发水 护发 沐浴 洗面奶 洁面 卫生巾 牙膏 牙刷 香水 脱毛 防晒 护肤 护理 精华 面膜 眼膜 滋润 补水 抗皱 紧致
食品饮料 美味 矿泉水 饮料 奶 汤 茶 咖啡 糖 食品 米饭 玉米 零食 肉 速食 即食 代餐 轻食 火锅 油 面 饼干 串 杧果 榴莲 鸡蛋 汤面 面包 水果 鲜果 解馋 美味
家居用品 家居 家用 家庭 厕所 卫生间 厨房 宿舍 除味 清洁 收纳 抽 卫生纸 面巾纸 洗脸 垃圾 梳子 水杯 杯 盘 湿巾 浴巾 洗衣 被子
服装饰品 衣服 上衣 内衣 卫衣 开衫 衬衫 T恤 短袖 长袖 外套 长裤 短裤 内裤 睡裤 短裙 长裙 连衣裙 家居服 羽绒服 浴袍 跑鞋 运动鞋 包 马甲 腰带 皮带 保暖 袜子 发饰
图书教育 图书 书籍 文学 教育 早教 课程 启蒙 益智 漫画 学生 笔 书包 课本 学习 知识 册
数码电子 智能 手机 计算机 耳机 相机 充电宝"""

#对上述字符串的每个关键词进行归类 关键词: 类别
dcatg = {}  #字典
catg = basic_data.split('\n')  #按换行符分割
for i in catg:
    main_cat = i.strip().split(' ')[0]  #第一个词是类别
    kw_cat = i.strip().split(' ')[1:]  #关键词列表
    for j in kw_cat:
        dcatg[j] = main_cat  #关键词与类别对应
print(dcatg)

#品类
type = []  #品类
for i in range(df.shape[0]):  #遍历每行数据
    exist = False
    for j in df['subtitle'][i]:  #遍历subtitle每行的元素
        if j in dcatg:
            type.append(dcatg[j])  #将关键词对应的类别取出
            exist = True
            break
    if not exist:
        type.append('其他')  #没有字典中的关键词的归为“其他”类
df['type'] = type
print(df[['type', 'subtitle']].head(0))

#品类占比统计
type_count = df['type'].value_counts(normalize=True) #统计每个品类占比
plt.pie(type_count,
        labels=type_count.index,  #品类标签
        autopct='%.1f%%',  #数据标注格式，保留一位小数
        labeldistance=1.23,  #标签标注位置与数据块径向距离的比率为1.23
        pctdistance=1.12)  #数据标注位置与数据块径向距离的比率为1.12
plt.show()


# =========================
# 第12页：热销商品分析
# =========================
# 统计销量top10商品情况，展示商品名称、价格、品类、销售量
top10 = df.sort_values(by='sale_count', ascending=False)[
    ['product_title', 'price', 'type', 'sale_count']
].head(10)

print(top10)


# =========================
# 第13页：价格效应分析
# =========================
# 统计不同价格等级商品的平均销售量，并绘制折线图
price_sale_mean = df.groupby('price_group')['sale_count'].mean()
price_sale_mean = price_sale_mean.reindex(['低价商品', '中价商品', '高价商品'])

plt.plot(price_sale_mean.index, price_sale_mean.values, marker='o')
plt.xlabel('价格等级')
plt.ylabel('平均销售量')
plt.show()


# =========================
# 第14页：品类效应分析（箱线图）
# =========================
#将商品品类与销量连接
type_concat = df[['type', 'sale_count']]

#不同品类商品销量
type_salecount_0 = type_concat.loc[type_concat['type'] == '美妆']['sale_count']
type_salecount_1 = type_concat.loc[type_concat['type'] == '个护']['sale_count']
type_salecount_2 = type_concat.loc[type_concat['type'] == '食品饮料']['sale_count']
type_salecount_3 = type_concat.loc[type_concat['type'] == '家居用品']['sale_count']
type_salecount_4 = type_concat.loc[type_concat['type'] == '服装饰品']['sale_count']
type_salecount_5 = type_concat.loc[type_concat['type'] == '图书教育']['sale_count']
type_salecount_6 = type_concat.loc[type_concat['type'] == '数码电子']['sale_count']
type_salecount_7 = type_concat.loc[type_concat['type'] == '其他']['sale_count']

#print(type_salecount_0, type_salecount_1, type_salecount_2, type_salecount_3, type_salecount_4, type_salecount_5, type_salecount_6, type_salecount_7)

#箱线图
plt.figure(figsize=(9, 6))
plt.boxplot(
    (type_salecount_0, type_salecount_1, type_salecount_2, type_salecount_3, type_salecount_4, type_salecount_5, type_salecount_6, type_salecount_7),
    labels=('美妆', '个护', '食品饮料', '家居用品', '服装饰品', '图书教育', '数码电子', '其他'),
    medianprops={'ls': '--'},  #中位线线型：虚线
    meanline=True, showmeans=True, meanprops={'ls': '-'}  #平均线线型：实线
)
plt.grid(True)  #显示网格
plt.xlabel('商品品类')
plt.ylabel('销量')
plt.show()
# =========================
# 第15页：品类效应分析（平均销售量和总销售量）
# =========================
#不同品类商品平均销量
type_salecount_0_mean = type_concat.loc[type_concat['type'] == '美妆']['sale_count'].mean()
type_salecount_1_mean = type_concat.loc[type_concat['type'] == '个护']['sale_count'].mean()
type_salecount_2_mean = type_concat.loc[type_concat['type'] == '食品饮料']['sale_count'].mean()
type_salecount_3_mean = type_concat.loc[type_concat['type'] == '家居用品']['sale_count'].mean()
type_salecount_4_mean = type_concat.loc[type_concat['type'] == '服装饰品']['sale_count'].mean()
type_salecount_5_mean = type_concat.loc[type_concat['type'] == '图书教育']['sale_count'].mean()
type_salecount_6_mean = type_concat.loc[type_concat['type'] == '数码电子']['sale_count'].mean()
type_salecount_7_mean = type_concat.loc[type_concat['type'] == '其他']['sale_count'].mean()

#print(type_salecount_0_mean, type_salecount_1_mean, type_salecount_2_mean, type_salecount_3_mean, type_salecount_4_mean, type_salecount_5_mean, type_salecount_6_mean, type_salecount_7_mean)

#不同品类商品总销量
type_salecount_0_sum = type_concat.loc[type_concat['type'] == '美妆']['sale_count'].sum()
type_salecount_1_sum = type_concat.loc[type_concat['type'] == '个护']['sale_count'].sum()
type_salecount_2_sum = type_concat.loc[type_concat['type'] == '食品饮料']['sale_count'].sum()
type_salecount_3_sum = type_concat.loc[type_concat['type'] == '家居用品']['sale_count'].sum()
type_salecount_4_sum = type_concat.loc[type_concat['type'] == '服装饰品']['sale_count'].sum()
type_salecount_5_sum = type_concat.loc[type_concat['type'] == '图书教育']['sale_count'].sum()
type_salecount_6_sum = type_concat.loc[type_concat['type'] == '数码电子']['sale_count'].sum()
type_salecount_7_sum = type_concat.loc[type_concat['type'] == '其他']['sale_count'].sum()

#柱状折线图
x = [1, 2, 3, 4, 5, 6, 7, 8]
y1 = [type_salecount_0_mean, type_salecount_1_mean, type_salecount_2_mean, type_salecount_3_mean, type_salecount_4_mean, type_salecount_5_mean, type_salecount_6_mean, type_salecount_7_mean]
y2 = [type_salecount_0_sum, type_salecount_1_sum, type_salecount_2_sum, type_salecount_3_sum, type_salecount_4_sum, type_salecount_5_sum, type_salecount_6_sum, type_salecount_7_sum]
x_label = ['美妆', '个护', '食品饮料', '家居用品', '服装饰品', '图书教育', '数码电子', '其他']

plt.figure(figsize=(7, 6))
plt.subplot(111)
#在主坐标轴绘制柱状图
plt.bar(x, y2, color='moccasin', label='总销量')
plt.legend(loc='upper left')

plt.xticks(x, x_label)
plt.xlabel('商品品类')

#在次坐标轴绘制折线图
plt.twinx()
plt.plot(x, y1, ls='--', lw=2, color='lightcoral', marker='*', ms=10, label='平均销量')
plt.legend(loc='upper right')

plt.xticks(x, x_label)
plt.show()
# =========================
# 第16页：带货效应分析
# =========================

# 根据博主的粉丝数将博主分为“小粉丝量博主”“中粉丝量博主”和“高粉丝量博主”
def fans_count_group(x):
    if x <= 100000:
        return '小粉丝量博主'
    elif x <= 1000000:
        return '中粉丝量博主'
    else:
        return '高粉丝量博主'

df['fans_group'] = df['fans_count'].apply(fans_count_group)

# 使用 concat() 函数将博主等级与销售量连接
fans_sale = pd.concat([df['fans_group'], df['sale_count']], axis=1)

# 使用 mean() 函数计算不同等级博主的平均带货销量
fans_sale_mean = fans_sale.groupby('fans_group')['sale_count'].mean()


# =========================
# 第17页：带货效应分析结果图
# =========================
#博主分类
def fans_count_group(fans_count):
    if fans_count <= 100000:
        return '小粉丝量博主'
    elif 100000 < fans_count <= 500000:
        return '中粉丝量博主'
    else:
        return '高粉丝量博主'

df['fans_count_group'] = df['fans_count'].apply(fans_count_group)
fans_concat = pd.concat([df['fans_count_group'], df['sale_count']], axis=1)  #将博主等级与销售量连接

#不同等级博主带货平均销售量
fans_salecount_0_mean = fans_concat.loc[fans_concat['fans_count_group'] == '小粉丝量博主']['sale_count'].mean()
fans_salecount_1_mean = fans_concat.loc[fans_concat['fans_count_group'] == '中粉丝量博主']['sale_count'].mean()
fans_salecount_2_mean = fans_concat.loc[fans_concat['fans_count_group'] == '高粉丝量博主']['sale_count'].mean()

#折线图
plt.figure(figsize=(9, 6))
x = [1, 2, 3]
y = [fans_salecount_0_mean, fans_salecount_1_mean, fans_salecount_2_mean]
x_label = ['小粉丝量博主', '中粉丝量博主', '高粉丝量博主']

plt.xticks(x, x_label)
plt.plot(x, y, label='平均销售量', color='lightcoral', marker='*', markersize=10)
plt.xlabel('粉丝量')
plt.ylabel('平均销售量/个')
plt.title('博主等级与销售量相关性分析')

# 标注数据
for a, b in zip(x, y):
    plt.text(a, b + 200, round(b))  # 数据显示的横坐标，位置高度，数据值保留整数

plt.legend()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
#将定性的品类特征转化为定量特征
def type_number(type):
    if type == '美妆':
        return 1  #美妆品类为 1
    elif type == '个护':
        return 2  #个护品类为 2
    elif type == '食品饮料':
        return 3  #食品饮料品类为 3
    elif type == '家居用品':
        return 4  #家居用品品类为 4
    elif type == '服装饰品':
        return 5  #服装饰品品类为 5
    elif type == '图书教育':
        return 6  #图书教育品类为 6
    elif type == '数码电子':
        return 7  #数码电子品类为 7
    else:
        return 8  #其他品类为 8

df['type_number'] = df['type'].apply(type_number)
X = df[['fans_count', 'price', 'type_number', 'likes']]  # 得到特征数据
Y = df['sale_count']  # 得到数据对应的销量

#划分训练集和测试集，测试集大小为 20%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=52)

#决策树回归
#初始化 DecisionTreeRegressor
rf_regressor = DecisionTreeRegressor(random_state=42)
# 在训练集上拟合模型
rf_regressor.fit(X_train, Y_train)
# 对测试集进行预测
Y_pred = rf_regressor.predict(X_test)
# 评估模型效果
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
r2 = r2_score(Y_test, Y_pred)
print("决策树回归")
print("均方根误差:", rmse)
print("R2 得分:", r2)

# k 折交叉验证
# 初始化多个回归模型
models = {
    'LinearRegression': LinearRegression(),
    'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42),
    'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42)
}
# 创建字典来保存模型分数以供比较
model_scores = {}

# 训练和交叉验证每个模型
for model_name, model in models.items():
    # 十次交叉验证
    nmse_scores = cross_val_score(model, X_train, Y_train, scoring='neg_mean_squared_error', cv=10)
    r2_scores = cross_val_score(model, X_train, Y_train, scoring='r2', cv=10)

    # 计算 RMSE
    rmse_scores = np.sqrt(-nmse_scores)

    # 计算不同模型 RMSE 和 r2 的均值
    model_scores[model_name] = {
        'RMSE Mean': np.mean(rmse_scores),
        'R2 Mean': np.mean(r2_scores)
    }
print(model_scores)
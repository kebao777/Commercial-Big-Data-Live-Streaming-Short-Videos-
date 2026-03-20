import warnings
import jieba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import re

# 模型与评估库
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    BaggingRegressor,
)
from sklearn.svm import SVR
import xgboost as xgb

# 全局设置（关键：关闭冗余警告）
warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.figsize"] = (10, 6)


# ========================= 1. 数据读取与基础探索 =========================
def load_and_explore_data(file_path):
    """读取数据并进行基础探索"""
    df = pd.read_excel(file_path)
    print("=" * 50)
    print("数据集基础信息")
    print("=" * 50)
    print(f"数据形状：{df.shape}")
    print(f"列名列表：{list(df.columns)}")
    print("\n数据类型：")
    print(df.dtypes)
    print("\n数值型字段描述统计：")
    print(df.select_dtypes(include=[np.number]).describe().round(2))
    return df


# ========================= 2. 特征工程（保持原样，不改数据处理方式） =========================
def feature_engineering(df):
    """特征工程：简化特征维度，避免冗余"""

    # 2.1 视频时长特征处理（兼容 1m23s / 49s / 01:23 / 83 等格式 → 总秒数）
    def convert_duration_to_seconds(duration_str):
        try:
            if pd.isna(duration_str):
                return 0

            s = str(duration_str).strip().lower()
            if s == "":
                return 0

            # 格式1：mm:ss
            if ":" in s:
                parts = s.split(":")
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    minutes = int(parts[0])
                    seconds = int(parts[1])
                    return minutes * 60 + seconds

            # 格式2：1m23s / 4m58s / 49s
            match = re.fullmatch(r"(?:(\d+)m)?(?:(\d+)s)?", s)
            if match:
                minutes = int(match.group(1)) if match.group(1) else 0
                seconds = int(match.group(2)) if match.group(2) else 0
                if minutes != 0 or seconds != 0:
                    return minutes * 60 + seconds

            # 格式3：纯数字，直接按秒处理
            if s.isdigit():
                return int(s)

            return 0
        except:
            return 0

    df["duration_seconds"] = df["duration"].apply(convert_duration_to_seconds)
    print("\nduration 示例解析：")
    print(df[["duration", "duration_seconds"]].head(10))
    # 简化时长分段（减少类别数）
    df["duration_segment"] = pd.cut(
        df["duration_seconds"],
        bins=[0, 60, 120, np.inf],
        labels=["0-60s", "61-120s", "120s+"],
        right=False
    )

    # 2.2 发布时间特征处理（简化）
    def parse_time(time_str):
        try:
            if pd.isna(time_str):
                return pd.NaT
            time_str = str(time_str).replace("/", "-")
            return pd.to_datetime(time_str)
        except:
            return pd.NaT

    df["publish_datetime"] = df["time"].apply(parse_time)
    df["publish_hour"] = df["publish_datetime"].dt.hour.fillna(df["publish_datetime"].dt.hour.median())
    # 简化小时分段
    df["publish_hour_segment"] = pd.cut(
        df["publish_hour"],
        bins=[0, 12, 18, 24],
        labels=["上午", "下午", "晚上"],
        right=False
    )
    df["is_workday"] = df["publish_datetime"].dt.weekday.apply(lambda x: 1 if x < 5 else 0).fillna(1)

    # 2.3 GPM特征优化（简化，减少异常值）
    def handle_outliers(data, col):
        q1 = data[col].quantile(0.05)
        q3 = data[col].quantile(0.95)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        lower_bound = q1 - 1.5 * iqr
        return data[col].clip(lower=lower_bound, upper=upper_bound)

    df["GPM_clean"] = handle_outliers(df, "GPM").fillna(df["GPM"].median())
    # 简化GPM分段
    df["GPM_level"] = pd.cut(
        df["GPM_clean"],
        bins=[-np.inf, 500, 2000, np.inf],
        labels=["低", "中", "高"],
        right=False
    )
    df["GPM_price_interact"] = df["GPM_clean"] * df["price"]

    # 2.4 核心衍生特征（仅保留高价值特征）
    df["like_fan_ratio"] = df["likes"] / (df["fans_count"] + 1)
    df["product_title_length"] = df["product_title"].apply(lambda x: len(str(x)) if pd.notna(x) else 0)

    # 2.5 商品品类特征优化（简化品类）
    df["product_title_clean"] = df["product_title"].astype(str).str.strip()
    stop_words = {"的", "了", "是", "在", "有", "和", "及", "等", "与", "或", "一个", "一款"}
    df["product_keywords"] = df["product_title_clean"].apply(
        lambda x: [word for word in jieba.cut(x) if word.strip() and word not in stop_words]
    )

    # 简化品类词典（减少品类数）
    category_dict = {
        "美妆": ["美妆", "化妆", "彩妆", "美容", "眉笔", "腮红", "眼影", "粉底液", "修容", "遮瑕", "口红", "睫毛",
                 "粉扑", "定妆", "面霜", "颈霜", "素颜霜", "卸妆"],
        "个护": ["个护", "洗发水", "护发", "沐浴", "洗面奶", "洁面", "卫生巾", "牙膏", "牙刷", "香水", "脱毛", "防晒",
                 "护肤", "护理", "精华", "面膜", "眼膜", "滋润", "补水", "抗皱", "紧致"],
        "食品饮料": ["食品饮料", "美味", "矿泉水", "饮料", "奶", "汤", "茶", "咖啡", "糖", "食品", "米饭", "玉米",
                     "零食", "肉", "速食", "即食", "代餐", "轻食", "火锅", "油", "面", "饼干", "串", "杧果", "榴莲",
                     "鸡蛋", "汤面", "面包", "水果", "鲜果", "解馋", "美味"],
        "家居用品": ["家居用品", "家居", "家用", "家庭", "厕所", "卫生间", "厨房", "宿舍", "除味", "清洁", "收纳", "抽",
                     "卫生纸", "面巾纸", "洗脸", "垃圾", "梳子", "水杯", "杯", "盘", "湿巾", "浴巾", "洗衣", "被子"],
        "服装饰品": ["服装饰品", "衣服", "上衣", "内衣", "卫衣", "开衫", "衬衫", "T恤", "短袖", "长袖", "外套", "长裤",
                     "短裤", "内裤", "睡裤", "短裙", "长裙", "连衣裙", "家居服", "羽绒服", "浴袍", "跑鞋", "运动鞋",
                     "包", "马甲", "腰带", "皮带", "保暖", "袜子", "发饰"],
        "图书教育": ["图书教育", "图书", "书籍", "文学", "教育", "早教", "课程", "启蒙", "益智", "漫画", "学生", "笔",
                     "书包", "课本", "学习", "知识", "册"],
        "数码电子": ["数码电子", "智能", "手机", "计算机", "耳机", "相机", "充电宝"]
    }

    def match_category(keywords):
        matched_cat = "其他"
        sorted_keywords = sorted(keywords, key=len, reverse=True)
        for word in sorted_keywords:
            for cat, cat_words in category_dict.items():
                if word in cat_words:
                    matched_cat = cat
                    return matched_cat
        return matched_cat

    df["product_category"] = df["product_keywords"].apply(match_category)

    print("\n" + "=" * 50)
    print("特征工程完成")
    print("=" * 50)
    print(f"最终数据形状：{df.shape}")
    return df


# ========================= 3. 建模数据准备（保持原样，不改数据处理方式） =========================
def prepare_model_data(df):
    """准备建模数据：简化特征，避免维度爆炸"""
    # 仅保留核心特征（减少特征数，解决分裂增益问题）
    feature_cols = [
        "fans_count", "likes", "price", "duration_seconds",
        "publish_hour", "is_workday", "GPM_clean", "like_fan_ratio",
        "product_category", "duration_segment", "publish_hour_segment", "GPM_level"
    ]
    target_col = "sale_count"

    # 数据清洗（更严格的过滤，保证数据质量）
    model_df = df[feature_cols + [target_col]].copy()
    model_df = model_df[(model_df[target_col] > 0) & (model_df[target_col] < model_df[target_col].quantile(0.99))]
    model_df = model_df.dropna()

    # 对数变换（仅对核心数值特征）
    log_features = ["fans_count", "likes", "price", "duration_seconds", "GPM_clean", target_col]
    for col in log_features:
        model_df[f"{col}_log"] = np.log1p(model_df[col])

    # 替换为log特征
    feature_cols = [col if col not in log_features else f"{col}_log" for col in feature_cols]
    target_col_log = f"{target_col}_log"

    # 分类特征编码（简化编码）
    categorical_features = ["product_category", "duration_segment", "publish_hour_segment", "GPM_level"]
    encoded_cats = pd.get_dummies(model_df[categorical_features], prefix=categorical_features, drop_first=True)

    # 构建特征矩阵
    numeric_features = [col for col in feature_cols if col not in categorical_features]
    X_numeric = model_df[numeric_features]
    X = pd.concat([X_numeric, encoded_cats], axis=1)
    y = model_df[target_col_log]

    # 数据划分（分层抽样，简化）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 数值特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[numeric_features])
    X_test_scaled = scaler.transform(X_test[numeric_features])

    # 合并特征
    X_train_final = pd.concat(
        [pd.DataFrame(X_train_scaled, columns=numeric_features, index=X_train.index),
         X_train[encoded_cats.columns]], axis=1
    )
    X_test_final = pd.concat(
        [pd.DataFrame(X_test_scaled, columns=numeric_features, index=X_test.index),
         X_test[encoded_cats.columns]], axis=1
    )

    print(f"建模数据集形状：{model_df.shape}")
    print(f"最终特征数：{X.shape[1]}个")
    print(f"训练集形状：{X_train_final.shape}")
    print(f"测试集形状：{X_test_final.shape}")
    return X_train_final, X_test_final, y_train, y_test, X, numeric_features


# ========================= 4. 模型训练与调优（仅保留指定模型） =========================
def train_and_tune_models(X_train, X_test, y_train, y_test):
    """模型训练：仅保留用户指定模型，并对指定模型做网格搜索调优"""

    # 基础模型：保留 LinearRegression、Lasso，以及其他待调优模型的基础版
    base_models = {
        "LinearRegression": LinearRegression(),
        "Lasso": Lasso(alpha=0.001, max_iter=5000),
        "DecisionTree": DecisionTreeRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(random_state=42, n_jobs=2),
        "ExtraTrees": ExtraTreesRegressor(random_state=42, n_jobs=2),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "Bagging": BaggingRegressor(random_state=42, n_jobs=2),
        "SVR": SVR(),
        "XGBoost": xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=2
        ),
    }

    def evaluate_base_models(models, X, y):
        results = []
        for name, model in models.items():
            r2_scores = cross_val_score(model, X, y, scoring="r2", cv=5)
            mse_scores = -cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5)
            rmse_scores = np.sqrt(mse_scores)
            results.append({
                "模型": name,
                "5折R²均值": round(r2_scores.mean(), 4),
                "5折MSE均值": round(mse_scores.mean(), 4),
                "5折RMSE均值": round(rmse_scores.mean(), 4),
            })
        return pd.DataFrame(results).sort_values("5折R²均值", ascending=False)

    base_results = evaluate_base_models(base_models, X_train, y_train)
    print("\n" + "=" * 50)
    print("基础模型5折交叉验证结果")
    print("=" * 50)
    print(base_results)

    # 各调优模型参数网格
    param_grids = {
        "DecisionTree_Tuned": {
            "max_depth": [3, 5, 7, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 3, 5, 10],
        },
        "RandomForest_Tuned": {
            "n_estimators": [80, 120],
            "max_depth": [5, 8, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 3, 5],
        },
        "ExtraTrees_Tuned": {
            "n_estimators": [80, 120],
            "max_depth": [5, 8, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 3, 5],
        },
        "GradientBoosting_Tuned": {
            "n_estimators": [80, 120],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [2, 3, 4],
            "subsample": [0.8, 1.0],
        },
        "Bagging_Tuned": {
            "n_estimators": [50, 80, 120],
            "max_samples": [0.8, 1.0],
            "max_features": [0.8, 1.0],
        },
        "SVR_Tuned": {
            "kernel": ["rbf"],
            "C": [1, 10, 30],
            "epsilon": [0.05, 0.1, 0.2],
            "gamma": ["scale", "auto"],
        },
        "XGBoost_Tuned": {
            "max_depth": [3, 5],
            "learning_rate": [0.03, 0.05, 0.1],
            "n_estimators": [50, 100, 150],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        },
    }

    estimators = {
        "DecisionTree_Tuned": DecisionTreeRegressor(random_state=42),
        "RandomForest_Tuned": RandomForestRegressor(random_state=42, n_jobs=2),
        "ExtraTrees_Tuned": ExtraTreesRegressor(random_state=42, n_jobs=2),
        "GradientBoosting_Tuned": GradientBoostingRegressor(random_state=42),
        "Bagging_Tuned": BaggingRegressor(random_state=42, n_jobs=2),
        "SVR_Tuned": SVR(),
        "XGBoost_Tuned": xgb.XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=2),
    }

    best_models = {}
    for model_name, estimator in estimators.items():
        print("\n" + "-" * 50)
        print(f"正在调优：{model_name}")
        print("-" * 50)
        grid = GridSearchCV(
            estimator=estimator,
            param_grid=param_grids[model_name],
            scoring="r2",
            cv=5,
            n_jobs=1,
            verbose=0,
        )
        grid.fit(X_train, y_train)
        best_models[model_name] = grid.best_estimator_
        print(f"最佳参数：{grid.best_params_}")
        print(f"最佳CV R²：{grid.best_score_:.4f}")

    # 最终模型集合：基础模型 + 调优模型
    final_models = {
        "LinearRegression": LinearRegression(),
        "Lasso": Lasso(alpha=0.001, max_iter=5000),
        **best_models,
    }

    def final_evaluate(models, X_train, X_test, y_train, y_test):
        results = []
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # 当前训练脚本的 y 是 log1p(sale_count)，这里先保留原有 log 尺度结果
            train_r2_log = r2_score(y_train, y_train_pred)
            test_r2_log = r2_score(y_test, y_test_pred)
            test_mse_log = mean_squared_error(y_test, y_test_pred)
            test_rmse_log = np.sqrt(test_mse_log)

            # 补充还原到原始销量尺度后的结果，便于后续统一解释
            y_test_raw = np.expm1(y_test)
            y_test_pred_raw = np.expm1(y_test_pred)
            test_mse_raw = mean_squared_error(y_test_raw, y_test_pred_raw)
            test_rmse_raw = np.sqrt(test_mse_raw)

            results.append({
                "模型": name,
                "训练集R²": round(train_r2_log, 4),
                "测试集R²": round(test_r2_log, 4),
                "测试集MSE": round(test_mse_log, 4),
                "测试集RMSE": round(test_rmse_log, 4)
            })
        return pd.DataFrame(results).sort_values("测试集R²", ascending=False)

    final_results = final_evaluate(final_models, X_train, X_test, y_train, y_test)
    print("\n" + "=" * 50)
    print("调优后模型最终评估结果")
    print("=" * 50)
    print(final_results)

    best_model_name = final_results.iloc[0]["模型"]
    best_model = final_models[best_model_name]
    print(f"\n最优模型：{best_model_name}（测试集R²={final_results.iloc[0]['测试集R²']:.4f}）")
    return best_model, final_results, best_models["XGBoost_Tuned"]


# ========================= 5. 可视化结果 =========================
def visualize_results(best_model, final_results, X_test, y_test, X):
    """可视化模型结果"""
    # 5.1 模型性能对比图
    plt.figure(figsize=(14, 7))
    models = final_results["模型"]
    test_r2 = final_results["测试集R²"]

    bars = plt.bar(models, test_r2, edgecolor="black", alpha=0.8)
    for bar, r2 in zip(bars, test_r2):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f"{r2:.4f}", ha='center', va='bottom', fontsize=9)

    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    plt.xlabel("模型", fontsize=12)
    plt.ylabel("测试集R方", fontsize=12)
    plt.title("各模型测试集R方对比（指定模型版）", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("模型R方对比图.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 5.2 最优模型预测对比图
    y_test_pred = best_model.predict(X_test)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.6, s=30)
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="完美预测线")

    plt.xlabel("真实销量", fontsize=12)
    plt.ylabel("预测销量", fontsize=12)
    plt.title(f"{best_model.__class__.__name__}模型：真实值vs预测值（R方={r2_score(y_test, y_test_pred):.4f}）",
              fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("最优模型预测对比图.png", dpi=300, bbox_inches='tight')
    plt.show()


# ========================= 6. 主函数 =========================
def main(file_path):
    """主函数：执行全流程建模"""
    df = load_and_explore_data(file_path)
    df = feature_engineering(df)
    X_train_final, X_test_final, y_train, y_test, X, numeric_features = prepare_model_data(df)
    best_model, final_results, best_xgb = train_and_tune_models(X_train_final, X_test_final, y_train, y_test)
    visualize_results(best_model, final_results, X_test_final, y_test, X)

    print("\n" + "=" * 50)
    print("建模流程全部完成！生成文件：")
    print("1. 模型R²对比图.png")
    print("2. 最优模型预测对比图.png")
    print("=" * 50)


# ========================= 运行入口 =========================
if __name__ == "__main__":
    DATA_FILE_PATH = "短视频营销数据.xlsx"
    main(DATA_FILE_PATH)
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, \
    AdaBoostClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)

# =========================
# 0. 基本设置
# =========================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

RANDOM_STATE = 42

# =========================
# 1. 读取数据
# =========================
file_path = "直播电商数据集.xlsx"
sheet_name = "E Comm"

df = pd.read_excel(file_path, sheet_name=sheet_name)

print("数据形状：", df.shape)
print("\n字段信息：")
print(df.info())

# =========================
# 2. 定义目标变量和特征
# =========================
target_col = "Churn"

# 预测任务建议用更多特征，不只用城市级别、婚姻状况
drop_cols = ["CustomerID", target_col]
feature_cols = [col for col in df.columns if col not in drop_cols]

X = df[feature_cols].copy()
y = df[target_col].copy()

print("\n目标变量分布：")
print(y.value_counts())
print("流失率：{:.2%}".format(y.mean()))

# =========================
# 3. 区分数值列和类别列
# =========================
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = [col for col in X.columns if col not in categorical_cols]

print("\n数值特征：", numeric_cols)
print("类别特征：", categorical_cols)

# =========================
# 4. 划分训练集和测试集
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\n训练集形状：", X_train.shape)
print("测试集形状：", X_test.shape)

# =========================
# 5. 预处理器
# =========================
# 给线性模型用：数值特征要标准化
numeric_transformer_linear = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 给树模型用：一般不需要标准化
numeric_transformer_tree = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor_linear = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer_linear, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

preprocessor_tree = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer_tree, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# =========================
# 6. 定义多个候选模型
# =========================
models = {
    "LogisticRegression": Pipeline(steps=[
        ("preprocessor", preprocessor_linear),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ]),

    "DecisionTree": Pipeline(steps=[
        ("preprocessor", preprocessor_tree),
        ("model", DecisionTreeClassifier(
            max_depth=5,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ]),

    "RandomForest": Pipeline(steps=[
        ("preprocessor", preprocessor_tree),
        ("model", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ]),

    "ExtraTrees": Pipeline(steps=[
        ("preprocessor", preprocessor_tree),
        ("model", ExtraTreesClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ]),

    "GradientBoosting": Pipeline(steps=[
        ("preprocessor", preprocessor_tree),
        ("model", GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ))
    ]),

    "AdaBoost": Pipeline(steps=[
        ("preprocessor", preprocessor_tree),
        ("model", AdaBoostClassifier(
            random_state=RANDOM_STATE
        ))
    ])
}

# =========================
# 7. 模型训练与测试集评估
# =========================
results = []

print("\n========== 各模型测试结果 ==========\n")

for model_name, pipeline in models.items():
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan

    results.append({
        "模型": model_name,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": auc
    })

    print(f"模型：{model_name}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"ROC_AUC  : {auc:.4f}")
    print("-" * 40)

results_df = pd.DataFrame(results).sort_values(by="ROC_AUC", ascending=False)

print("\n========== 模型对比汇总 ==========")
print(results_df)

# =========================
# 8. 可视化模型比较
# =========================
plt.figure(figsize=(10, 5))
plt.bar(results_df["模型"], results_df["ROC_AUC"], edgecolor="gray")
plt.title("不同模型 ROC-AUC 对比")
plt.ylabel("ROC-AUC")
plt.xticks(rotation=20)

for i, v in enumerate(results_df["ROC_AUC"]):
    plt.text(i, v, f"{v:.4f}", ha="center", va="bottom")

plt.show()

# =========================
# 9. 选择最优模型
# =========================
best_model_name = results_df.iloc[0]["模型"]
print(f"\n最优基线模型（按 ROC-AUC 排序）: {best_model_name}")

best_pipeline = models[best_model_name]

# =========================
# 10. 对最优模型做参数优化
# =========================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

if best_model_name == "RandomForest":
    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 5, 10, 15],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4]
    }
elif best_model_name == "ExtraTrees":
    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 5, 10, 15],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4]
    }
elif best_model_name == "GradientBoosting":
    param_grid = {
        "model__n_estimators": [100, 150, 200],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [2, 3, 4]
    }
elif best_model_name == "LogisticRegression":
    param_grid = {
        "model__C": [0.01, 0.1, 1, 10],
        "model__solver": ["lbfgs", "liblinear"]
    }
else:
    # 其他模型如果被选为最优，给一个通用参数网格
    param_grid = {}

if len(param_grid) > 0:
    grid_search = GridSearchCV(
        estimator=best_pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print("\n========== 最优参数 ==========")
    print(grid_search.best_params_)
    print("最优交叉验证 ROC-AUC：{:.4f}".format(grid_search.best_score_))

    final_model = grid_search.best_estimator_
else:
    print("\n该模型未设置参数网格，直接使用基线模型。")
    final_model = best_pipeline.fit(X_train, y_train)

# =========================
# 11. 最优模型最终测试评估
# =========================
y_pred_final = final_model.predict(X_test)
y_prob_final = final_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred_final)
precision = precision_score(y_test, y_pred_final, zero_division=0)
recall = recall_score(y_test, y_pred_final, zero_division=0)
f1 = f1_score(y_test, y_pred_final, zero_division=0)
auc = roc_auc_score(y_test, y_prob_final)

print("\n========== 最优模型最终测试结果 ==========")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1       : {f1:.4f}")
print(f"ROC_AUC  : {auc:.4f}")

print("\n分类报告：")
print(classification_report(y_test, y_pred_final, digits=4, zero_division=0))

# =========================
# 12. 混淆矩阵
# =========================
cm = confusion_matrix(y_test, y_pred_final)

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["不流失", "流失"])
disp.plot(ax=ax, cmap="Blues", values_format="d")
plt.title(f"{best_model_name} 最优模型混淆矩阵")
plt.show()

# =========================
# 13. 特征重要性（树模型）或系数（逻辑回归）
# =========================
model_obj = final_model.named_steps["model"]
preprocessor_obj = final_model.named_steps["preprocessor"]

# 取出经过 one-hot 后的全部特征名
feature_names = preprocessor_obj.get_feature_names_out()

if hasattr(model_obj, "feature_importances_"):
    importance_df = pd.DataFrame({
        "特征": feature_names,
        "重要性": model_obj.feature_importances_
    }).sort_values(by="重要性", ascending=False)

    print("\n========== 特征重要性 Top 20 ==========")
    print(importance_df.head(20))

    plt.figure(figsize=(10, 6))
    top_n = 15
    top_df = importance_df.head(top_n).iloc[::-1]
    plt.barh(top_df["特征"], top_df["重要性"], edgecolor="gray")
    plt.title(f"{best_model_name} 特征重要性 Top {top_n}")
    plt.xlabel("重要性")
    plt.show()

elif hasattr(model_obj, "coef_"):
    coef_df = pd.DataFrame({
        "特征": feature_names,
        "系数": model_obj.coef_[0]
    }).sort_values(by="系数", ascending=False)

    print("\n========== 回归系数 Top 20 ==========")
    print(coef_df.head(20))

    plt.figure(figsize=(10, 6))
    top_n = 15
    top_df = coef_df.head(top_n).iloc[::-1]
    plt.barh(top_df["特征"], top_df["系数"], edgecolor="gray")
    plt.title(f"{best_model_name} 回归系数 Top {top_n}")
    plt.xlabel("系数")
    plt.show()

# =========================
# 14. 输出预测概率最高的高风险用户
# =========================
result_df = X_test.copy()
result_df["真实值"] = y_test.values
result_df["预测值"] = y_pred_final
result_df["流失概率"] = y_prob_final

result_df = result_df.sort_values(by="流失概率", ascending=False)

print("\n========== 高风险用户前10个 ==========")
print(result_df.head(10))
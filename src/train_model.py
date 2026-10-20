import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.metrics import confusion_matrix, classification_report
from data_preprocessing import prepare_data

# 获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 读取设备数据
DATA_PATH = PROJECT_ROOT / "data" / "ai4i2020.csv"
df = pd.read_csv(DATA_PATH)

# 定义输入特征
feature_columns = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# 定义输入 X 和标签 y
X = df[feature_columns]
y = df["Machine failure"]

print("===== 输入特征 X =====")
print(X.head())

print("\nX 的形状：", X.shape)
print("y 的形状：", y.shape)

print("\n===== 标签分布 =====")
print(y.value_counts())


# 将数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 从原来的训练集中划分验证集
X_train_new, X_val, y_train_new, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.25,
    random_state=42,
    stratify=y_train
)

print("\n===== 三部分数据集划分 =====")

print("训练集：", X_train_new.shape)
print("验证集：", X_val.shape)
print("测试集：", X_test.shape)

print("\n===== 各数据集故障率 =====")

print(f"训练集：{y_train_new.mean():.2%}")
print(f"验证集：{y_val.mean():.2%}")
print(f"测试集：{y_test.mean():.2%}")


print("\n===== 数据集划分结果 =====")

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

print("\n===== 故障比例 =====")
print(f"原始数据故障率：{y.mean():.2%}")
print(f"训练集故障率：{y_train.mean():.2%}")
print(f"测试集故障率：{y_test.mean():.2%}")



# 类别特征
categorical_features = ["Type"]

# 数值特征
numerical_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# 创建统一预处理器
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "num",
            StandardScaler(),
            numerical_features
        )
    ]
)

# 在训练集上拟合预处理器
X_train_processed = preprocessor.fit_transform(X_train)

# 对测试集应用同一套预处理规则
X_test_processed = preprocessor.transform(X_test)

print("\n===== 数据预处理结果 =====")
print("训练集形状：", X_train_processed.shape)
print("测试集形状：", X_test_processed.shape)

print("\n处理后的前3条数据：")
print(X_train_processed[:3])

# 创建完整机器学习流水线
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

# 训练模型
model.fit(X_train, y_train)

print("\n===== 逻辑回归模型训练完成 =====")

# 测试集预测
y_pred = model.predict(X_test)

# 计算评价指标
print("\n===== 测试集评价指标 =====")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred, zero_division=0):.4f}")

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("\n===== 混淆矩阵 =====")
print(cm)

# 输出完整分类报告
print("\n===== 分类报告 =====")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Normal", "Failure"],
    zero_division=0
))

# ===== 第三十五步：class_weight 对比实验 =====

experiments = {
    "Normal LR": None,
    "Balanced LR": "balanced"
}

print("\n===== class_weight 对比实验（验证集）=====")

for name, weight in experiments.items():

    model_exp = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            class_weight=weight,
            max_iter=1000,
            random_state=42
        ))
    ])

    # 只在6000条新训练样本上训练
    model_exp.fit(X_train_new, y_train_new)

    # 使用2000条验证样本评估
    y_val_pred = model_exp.predict(X_val)

    print(f"\n模型：{name}")

    print(
        f"Accuracy : "
        f"{accuracy_score(y_val, y_val_pred):.4f}"
    )

    print(
        f"Precision: "
        f"{precision_score(y_val, y_val_pred, zero_division=0):.4f}"
    )

    print(
        f"Recall   : "
        f"{recall_score(y_val, y_val_pred, zero_division=0):.4f}"
    )

    print(
        f"F1-score : "
        f"{f1_score(y_val, y_val_pred, zero_division=0):.4f}"
    )

    print("混淆矩阵：")
    print(confusion_matrix(y_val, y_val_pred, labels=[0, 1]))

# ===== 第三十六步：分类阈值对比实验 =====

# 创建普通逻辑回归模型
threshold_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        class_weight=None,
        max_iter=1000,
        random_state=42
    ))
])

# 仅使用训练集训练
threshold_model.fit(X_train_new, y_train_new)

# 获取验证集的故障预测概率
y_val_proba = threshold_model.predict_proba(X_val)[:, 1]

# 设置需要测试的分类阈值
thresholds = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

print("\n===== 分类阈值对比实验（验证集）=====")

for threshold in thresholds:

    # 根据阈值将概率转换为0或1
    y_pred_threshold = (
        y_val_proba >= threshold
    ).astype(int)

    precision = precision_score(
        y_val, y_pred_threshold, zero_division=0
    )

    recall = recall_score(
        y_val, y_pred_threshold, zero_division=0
    )

    f1 = f1_score(
        y_val, y_pred_threshold, zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_val, y_pred_threshold, labels=[0, 1]
    ).ravel()

    print(
        f"阈值={threshold:.2f} | "
        f"Precision={precision:.4f} | "
        f"Recall={recall:.4f} | "
        f"F1={f1:.4f} | "
        f"FP={fp} | FN={fn}"
    )

# ===== 第三十七步：保存阈值实验结果 =====

results = []

for threshold in thresholds:
    y_pred_threshold = (y_val_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val, y_pred_threshold, labels=[0, 1]
    ).ravel()

    results.append({
        "threshold": threshold,
        "precision": precision_score(
            y_val, y_pred_threshold, zero_division=0
        ),
        "recall": recall_score(
            y_val, y_pred_threshold, zero_division=0
        ),
        "f1": f1_score(
            y_val, y_pred_threshold, zero_division=0
        ),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn
    })

# 转换为 DataFrame
results_df = pd.DataFrame(results)

# 创建实验结果目录
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 保存 CSV
save_path = RESULTS_DIR / "lr_threshold_validation.csv"
results_df.to_csv(save_path, index=False)

print("\n===== 阈值实验结果已保存 =====")
print(save_path)
print(results_df.round(4))

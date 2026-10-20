from data_preprocessing import prepare_data
from evaluate_model import evaluate_model

import joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# 1. 调用独立的数据处理模块
X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()

# 2. 定义类别特征与数值特征
categorical_features = ["Type"]

numerical_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# 3. 创建预处理器
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"),
     categorical_features),
    ("num", StandardScaler(), numerical_features)
])

# 4. 创建完整模型流水线
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

# 5. 训练模型
model.fit(X_train, y_train)

print("===== 模块化逻辑回归训练完成 =====")
print("训练集形状：", X_train.shape)
print("验证集形状：", X_val.shape)

# 6. 在验证集上预测
y_val_pred = model.predict(X_val)

print("预测结果形状：", y_val_pred.shape)
print("预测为故障的样本数：", (y_val_pred == 1).sum())

# 使用统一评估函数
metrics = evaluate_model(
    y_val,
    y_val_pred,
    model_name="Logistic Regression"
)

# ===== 第四十二步：保存训练好的模型 =====

# 获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 定义模型保存目录
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 定义保存路径
MODEL_PATH = MODEL_DIR / "logistic_regression.joblib"

# 保存完整 Pipeline
joblib.dump(model, MODEL_PATH)

print("\n===== 模型保存成功 =====")
print("模型路径：", MODEL_PATH)

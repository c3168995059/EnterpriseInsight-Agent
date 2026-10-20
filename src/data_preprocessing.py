import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "ai4i2020.csv"


def load_data():
    """加载工业设备数据集。"""
    df = pd.read_csv(DATA_PATH)
    return df


def prepare_data():
    """划分训练集、验证集和测试集。"""

    df = load_data()

    feature_columns = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    X = df[feature_columns]
    y = df["Machine failure"]

    # 第一次划分：80%训练，20%测试
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 第二次划分：从训练部分取25%作为验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        random_state=42,
        stratify=y_train
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    data = prepare_data()

    for name, item in zip(
        ["X_train", "X_val", "X_test",
         "y_train", "y_val", "y_test"],
        data
    ):
        print(name, item.shape)

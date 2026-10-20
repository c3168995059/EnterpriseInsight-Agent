
import pandas as pd
from pathlib import Path

# 1. 获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 2. 构建数据集路径
DATA_PATH = PROJECT_ROOT / "data" / "ai4i2020.csv"

# 3. 读取数据
df = pd.read_csv(DATA_PATH)

# 4. 查看前5行
print("===== 数据集前5行 =====")
print(df.head())

# 5. 查看数据集大小
print("\n===== 数据集大小 =====")
print("行数：", df.shape[0])
print("列数：", df.shape[1])

# 6. 查看列名
print("\n===== 数据集列名 =====")
print(df.columns.tolist())

# 7. 查看数据集基本信息
print("\n===== 数据集信息 =====")
df.info()

# 8. 统计每一列的缺失值
print("\n===== 缺失值统计 =====")
print(df.isnull().sum())

# 9. 统计设备故障样本
print("\n===== 故障样本分布 =====")
print(df["Machine failure"].value_counts())

# 10. 统计正常和故障样本数量
normal_count = (df["Machine failure"] == 0).sum()
failure_count = (df["Machine failure"] == 1).sum()

# 11. 计算故障率
total_count = len(df)
failure_rate = failure_count / total_count

print("\n===== 设备故障统计 =====")
print("正常样本数量：", normal_count)
print("故障样本数量：", failure_count)
print(f"设备故障率：{failure_rate:.2%}")

# 12. 假设模型将全部样本预测为正常
baseline_accuracy = normal_count / total_count
print(f"全部预测正常时的准确率：{baseline_accuracy:.2%}")

# 13. 按产品类型分组，统计故障情况
type_stats = df.groupby("Type")["Machine failure"].agg(
    ["count", "sum", "mean"]
)

# 14. 修改统计结果的列名
type_stats.columns = [
    "样本数量",
    "故障数量",
    "故障率"
]

# 15. 将故障率转换为百分数
type_stats["故障率"] = type_stats["故障率"] * 100

print("\n===== 不同产品类型的故障统计 =====")
print(type_stats.round(2))

# 16. 筛选 L 类且发生故障的记录
l_failure = df[
    (df["Type"] == "L") &
    (df["Machine failure"] == 1)
]

print("\n===== L 类产品故障记录 =====")
print(l_failure.head())

print("\nL 类产品故障记录数量：", len(l_failure))

# 17. 选择需要分析的设备特征
features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# 18. 按是否故障分组，计算各特征的平均值
feature_stats = df.groupby("Machine failure")[features].mean()

print("\n===== 正常与故障样本的特征均值 =====")
print(feature_stats.round(2))

# 19. 分别查看正常组和故障组的扭矩分布
normal_torque = df.loc[
    df["Machine failure"] == 0,
    "Torque [Nm]"
]

failure_torque = df.loc[
    df["Machine failure"] == 1,
    "Torque [Nm]"
]

print("\n===== 正常样本扭矩统计 =====")
print(normal_torque.describe())

print("\n===== 故障样本扭矩统计 =====")
print(failure_torque.describe())

import matplotlib.pyplot as plt
import seaborn as sns

# 设置图表风格
sns.set_theme(style="whitegrid")

# 创建画布
plt.figure(figsize=(8, 5))

# 绘制箱线图
sns.boxplot(
    data=df,
    x="Machine failure",
    y="Tool wear [min]"
)

# 设置标题和坐标轴
plt.title("Tool Wear Distribution by Machine Failure")
plt.xlabel("Machine Failure (0=Normal, 1=Failure)")
plt.ylabel("Tool wear [min]")

# 保存图表
save_path = PROJECT_ROOT / "figures" / "tool_wear_boxplot.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")

# 显示图表
plt.show()

print("图表已保存至：", save_path)

# 20. 选择用于相关性分析的特征
corr_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure"
]

# 21. 计算相关系数矩阵
corr_matrix = df[corr_features].corr()

print("\n===== 特征相关系数矩阵 =====")
print(corr_matrix.round(3))

# 22. 绘制特征相关性热力图
plt.figure(figsize=(10, 8))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.title("Feature Correlation Heatmap")

# 调整布局
plt.tight_layout()

# 保存图表
save_path = PROJECT_ROOT / "figures" / "correlation_heatmap.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")

# 显示图表
plt.show()

print("相关性热力图已保存至：", save_path)
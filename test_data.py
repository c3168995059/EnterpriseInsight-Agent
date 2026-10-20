import numpy as np
import pandas as pd

print("NumPy 版本：", np.__version__)
print("Pandas 版本：", pd.__version__)

# 模拟三台工业设备的运行数据
data = {
    "设备编号": ["A001", "A002", "A003"],
    "温度": [65, 82, 71],
    "振动": [0.12, 0.35, 0.18],
}

# 创建 DataFrame
df = pd.DataFrame(data)

print("\n设备运行数据：")
print(df)

print("\n平均温度：", np.mean(df["温度"]))
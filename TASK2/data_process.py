import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 加载数据与描述性统计
# ==========================================
# 使用相对路径加载数据集
file_path = './heart_failure_clinical_records_dataset.csv'
df = pd.read_csv(file_path)

print("=== 描述性统计分析 ===")
# 打印核心连续变量的统计信息
continuous_cols = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium', 'time']
print(df[continuous_cols].describe().round(2))

# ==========================================
# 2. 缺失值与异常值检查
# ==========================================
print("\n=== 缺失值检查 ===")
missing_values = df.isnull().sum()
if missing_values.sum() == 0:
    print("结论：数据集中不存在缺失值。")
else:
    print(missing_values[missing_values > 0])

print("\n=== 异常值分析与处理 (Clipping) ===")
# 临床数据中，极高/极低值通常代表危重病理状态（如极高的血清肌酐或 CPK），直接删除会损失预测死亡事件的关键信息。
# 妥协方案：对极度偏离的长尾数据使用 1% 和 99% 分位数进行截断（Winsorization/Clipping）。
cols_to_clip = ['creatinine_phosphokinase', 'serum_creatinine', 'platelets']

for col in cols_to_clip:
    lower_bound = df[col].quantile(0.01)
    upper_bound = df[col].quantile(0.99)
    df[col] = np.clip(df[col], lower_bound, upper_bound)
print(f"已对 {cols_to_clip} 的极值(1% - 99%)进行截断处理，保留了病理特征同时削弱了离群点对模型梯度的干扰。")

# ==========================================
# 3. 数据标准化/归一化
# ==========================================
# 提取特征与标签
X = df.drop(columns=['DEATH_EVENT'])
y = df['DEATH_EVENT']

# 分离连续变量和分类变量（二态变量不需要标准化）
categorical_cols = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']

# 初始化标准化器
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[continuous_cols] = scaler.fit_transform(X[continuous_cols])

print("\n=== 数据标准化完成 ===")
print("连续变量已转化为均值为 0，标准差为 1 的分布。")
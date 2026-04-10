import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestClassifier

# 加载处理后的数据（假设 df 是上一阶段处理好的数据）
# 为了演示，我们直接读取原始数据并进行简单处理
df = pd.read_csv('./heart_failure_clinical_records_dataset.csv')

# ==========================================
# 1. 相关性热力图 (Correlation Heatmap)
# ==========================================
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# ==========================================
# 2. 统计学检验 (T-Test) 
# ==========================================
# 比较死亡组与存活组在连续变量上的均值差异
print("=== T-检验结果 (P-value < 0.05 表示具有显著差异) ===")
continuous_features = ['age', 'ejection_fraction', 'serum_creatinine', 'serum_sodium', 'time']
for feat in continuous_features:
    group0 = df[df['DEATH_EVENT'] == 0][feat]
    group1 = df[df['DEATH_EVENT'] == 1][feat]
    t_stat, p_val = stats.ttest_ind(group0, group1)
    print(f"{feat:20}: p-value = {p_val:.4f}")

# ==========================================
# 3. 基于随机森林的特征重要性 (Feature Importance)
# ==========================================
X = df.drop(columns=['DEATH_EVENT'])
y = df['DEATH_EVENT']

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# 提取重要性并排序
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

print("\n=== 随机森林特征重要性排名前五 ===")
print(importances.head(5))
#排名第一：随访时间 (Time)统计表现： 与 DEATH_EVENT 呈现极强的负相关（相关系数约 -0.53）。
# 医学解释： 这反映了心力衰竭的病程进展。随访时间越短且发生事件，说明患者在早期即出现急性恶化。在预测模型中，这通常是最强的判别因素。

# 排名第二：射血分数 (Ejection Fraction)统计表现： 死亡组的平均射血分数显著低于存活组 ($p < 0.001$)。
# 医学解释： 射血分数衡量心脏每搏动一次泵出的血液百分比。心脏泵血能力越弱（低 EF 值），全身器官供血越不足，是心力衰竭严重程度的核心金标准。

# 排名第三：血清肌酐 (Serum Creatinine)统计表现： 死亡组的血清肌酐水平明显升高。
# 医学解释： 血清肌酐是衡量肾功能的关键指标。心脏衰竭常导致“心肾综合征”，即心脏排血不足导致肾脏受损。高肌酐水平预示着多器官功能衰竭风险增加。

# 可视化
plt.figure(figsize=(10, 6))
importances.plot(kind='barh')
plt.title("Feature Importance by Random Forest")
plt.show()
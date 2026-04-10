import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# 1. 数据准备
df = pd.read_csv('./heart_failure_clinical_records_dataset.csv')
X = df.drop(columns=['DEATH_EVENT'])
y = df['DEATH_EVENT']

# 划分训练集与测试集 (80% 训练, 20% 测试)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化 (对逻辑回归和 MLP 至关重要)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. 模型定义
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "MLP (Neural Network)": MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42)
}

# 3. 训练与评估
results = {}
plt.figure(figsize=(10, 8))

for name, model in models.items():
    # 训练
    model.fit(X_train_scaled, y_train)
    
    # 预测
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # 计算指标
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "AUC": roc_auc
    }
    
    # 绘制 ROC 曲线
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

# 4. 输出评估表格
print("\n=== 模型性能对比 ===")
report = pd.DataFrame(results).T
print(report.round(3))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend()
plt.show()

# ==========================================
# 5. 新患者预测示例 (Inference)
# ==========================================
# 假设一位新患者：65岁, 有贫血, CPK 582, 无糖尿病, 射血分数 20%, 有高血压, 
# 血小板 265000, 血清肌酐 1.9, 血清钠 130, 男性, 不吸烟, 随访 4 天
new_patient = np.array([[65, 1, 582, 0, 20, 1, 265000, 1.9, 130, 1, 0, 4]])
new_patient_scaled = scaler.transform(new_patient)

# 使用表现通常最好的 XGBoost 进行预测
prob = models["XGBoost"].predict_proba(new_patient_scaled)[0][1]

print("\n=== 新患者预测结果 ===")
print(f"该患者的临床特征：年龄65, 射血分数20%, 血清肌酐1.9...")
print(f"预测死亡风险概率: {prob:.2%}")
if prob > 0.5:
    print("结论：该患者属于高危群体，建议加强临床监护。")
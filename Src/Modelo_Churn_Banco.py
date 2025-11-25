import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

df = pd.read_csv('/datasets/Churn.csv')

print(df.info())
print(df.head())
print(df.isnull().sum())

df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
df['Tenure'] = df['Tenure'].fillna(-1)

df.info()
df.head()

df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

df.head()
df.info()

features = df.drop('Exited', axis=1) #Todas menos la columna objetivo
target = df['Exited'] #Solo la columna objetivo

features_train, features_temp, target_train, target_temp = train_test_split(features, target, test_size=0.4, random_state=12345, stratify=target)

features_valid, features_test, target_valid, target_test = train_test_split(features_temp, target_temp, test_size=0.5, random_state=12345, stratify=target_temp)

print('Train:', features_train.shape, target_train.shape)
print('Validation:', features_valid.shape, target_valid.shape)
print('Test:', features_test.shape, target_test.shape)

class_counts = target_train.value_counts()
class_percent = target_train.value_counts(normalize=True)*100

print('Conteo de clases:\n', class_counts)
print('\nPorcentaje de clases:\n', class_percent)

model = LogisticRegression(random_state=12345)
model.fit(features_train, target_train)
target_pred = model.predict(features_valid)

accuracy = accuracy_score(target_valid, target_pred)
precision = precision_score(target_valid, target_pred)
recall = recall_score(target_valid, target_pred)
f1 =f1_score(target_valid, target_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

rf_model = RandomForestClassifier(random_state=12345, n_estimators=100)
rf_model.fit(features_train, target_train)
rf_target_predict = rf_model.predict(features_valid)

accuracy = accuracy_score(target_valid, rf_target_predict)
precision = precision_score(target_valid, rf_target_predict)
recall = recall_score(target_valid, rf_target_predict)
f1 = f1_score(target_valid, rf_target_predict)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

model_balanced = LogisticRegression(class_weight='balanced', random_state=12345)
model_balanced.fit(features_train, target_train)
model_balanced_pred = model_balanced.predict(features_valid)

model_balanced_acc = accuracy_score(target_valid, model_balanced_pred)
model_balanced_prec = precision_score(target_valid, model_balanced_pred)
model_balanced_rec = recall_score(target_valid, model_balanced_pred)
model_balanced_f1 = f1_score(target_valid, model_balanced_pred)

print(f"Accuracy: {model_balanced_acc:.4f}")
print(f"Precision: {model_balanced_prec:.4f}")
print(f"Recall: {model_balanced_rec:.4f}")
print(f"F1 Score: {model_balanced_f1:.4f}")

rf_model_balanced = RandomForestClassifier(class_weight='balanced', random_state=12345, n_estimators=100)
rf_model_balanced.fit(features_train, target_train)
rf_model_balanced_pred = rf_model_balanced.predict(features_valid)

rf_model_balanced_acc = accuracy_score(target_valid, rf_model_balanced_pred)
rf_model_balanced_prec = precision_score(target_valid, rf_model_balanced_pred)
rf_model_balanced_rec = recall_score(target_valid, rf_model_balanced_pred)
rf_model_balanced_f1 = f1_score(target_valid, rf_model_balanced_pred)

print(f"Accuracy: {rf_model_balanced_acc:.4f}")
print(f"Precision: {rf_model_balanced_prec:.4f}")
print(f"Recall: {rf_model_balanced_rec:.4f}")
print(f"F1 Score: {rf_model_balanced_f1:.4f}")

results = {
    "Modelo": [
        "Logistic Regression (sin balance)",
        "Random Forest (sin balance)",
        "Logistic Regression (balanceado)",
        "Random Forest (balanceado)"
    ],
    "Accuracy": [
        0.7875,  # LR sin balance
        0.8640,  # RF sin balance
        model_balanced_acc,
        rf_model_balanced_acc
    ],
    "Precision": [
        0.3111,  # LR sin balance
        0.7595,  # RF sin balance
        model_balanced_prec,
        rf_model_balanced_prec
    ],
    "Recall": [
        0.0343,  # LR sin balance
        0.4877,  # RF sin balance
        model_balanced_rec,
        rf_model_balanced_rec
    ],
    "F1 Score": [
        0.0618,  # LR sin balance
        0.5940,  # RF sin balance
        model_balanced_f1,
        rf_model_balanced_f1
    ]
}

df_results = pd.DataFrame(results)
print(df_results)

target_proba = model_balanced.predict_proba(features_test)[:, 1]
target_proba_rf = rf_model_balanced.predict_proba(features_test)[:, 1]

fpr_lr, tpr_lr, _ = roc_curve(target_test, target_proba)
fpr_rf, tpr_rf, _ = roc_curve(target_test, target_proba_rf)

auc_lr = roc_auc_score(target_test, target_proba)
auc_rf = roc_auc_score(target_test, target_proba_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {auc_lr:.3f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')

plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curvas ROC - Modelos Balanceados')
plt.legend()
plt.grid(True)
plt.show()
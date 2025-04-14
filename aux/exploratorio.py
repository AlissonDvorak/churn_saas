import pandas as pd
from pycaret.classification import setup, compare_models, save_model, pull
from imblearn.over_sampling import SMOTE

# ========================
# CODIGO EXPLORATÓRIO
# ========================


# Carregar e pré-processar
df = pd.read_pickle("processed/cleaned_data/cleaned_data.pkl")
df.drop(columns=['customer_id'], inplace=True)

# One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['company_size'], drop_first=True)

# ========================
# 1) UNDERSAMPLING SETUP
# ========================
df_majority = df[df['churn'] == 0]
df_minority = df[df['churn'] == 1]

frac = len(df_minority) / len(df_majority)
df_majority_balanced = (
    df_majority
    .groupby('company_size')
    .apply(lambda x: x.sample(frac=frac, random_state=42))
    .reset_index(drop=True)
)
df_under = pd.concat([df_majority_balanced, df_minority], axis=0).sample(frac=1, random_state=42)
df_under_encoded = pd.get_dummies(df_under, columns=['company_size'], drop_first=True)

print("🧪 Comparando modelos com PyCaret (Undersampling)...")
setup(data=df_under_encoded, target='churn', train_size=0.8, index=False, session_id=42, fix_imbalance=False, verbose=False)
best_under = compare_models(sort='Recall')
save_model(best_under, 'best_model_undersampling')
results_under = pull()

# ========================
# 2) SMOTE SETUP
# ========================
X = df_encoded.drop(columns=['churn'])
y = df_encoded['churn']
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)
df_smote = pd.concat([pd.DataFrame(X_smote, columns=X.columns), pd.Series(y_smote, name='churn')], axis=1)

print("\n🧪 Comparando modelos com PyCaret (SMOTE)...")
setup(data=df_smote, target='churn', train_size=0.8, session_id=42, index=False, fix_imbalance=False, verbose=False)
best_smote = compare_models(sort='Recall')
save_model(best_smote, 'best_model_smote')
results_smote = pull()

# ========================
# 3) Resultados
# ========================
print("\n📊 Melhores modelos com Undersampling:")
print(results_under.head())
print(f"🔍 Modelo campeão com undersampling: {type(best_under).__name__}")

print("\n📊 Melhores modelos com SMOTE (Oversampling):")
print(results_smote.head())
print(f"🔍 Modelo campeão com SMOTE: {type(best_smote).__name__}")

# ======================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Carrega os dados
df = pd.read_pickle("processed/cleaned_data/cleaned_data.pkl")
df.drop(columns=['customer_id'], inplace=True)

# Undersampling estratificado
df_majority = df[df['churn'] == 0]
df_minority = df[df['churn'] == 1]
frac = len(df_minority) / len(df_majority)
df_majority_balanced = (
    df_majority.groupby('company_size')
    .apply(lambda x: x.sample(frac=frac, random_state=42))
    .reset_index(drop=True)
)
df_under = pd.concat([df_majority_balanced, df_minority], axis=0).sample(frac=1, random_state=42)
df_under_encoded = pd.get_dummies(df_under, columns=['company_size'], drop_first=True)

# Divide undersampling
X_under = df_under_encoded.drop(columns=['churn'])
y_under = df_under_encoded['churn']
from sklearn.model_selection import train_test_split
X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(X_under, y_under, test_size=0.2, random_state=42)

# Dataset original
df_encoded = pd.get_dummies(df, columns=['company_size'], drop_first=True)
X_orig = df_encoded.drop(columns=['churn'])
y_orig = df_encoded['churn']
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X_orig, y_orig, test_size=0.2, random_state=42)

# RF com peso ajustado
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight={0: 2, 1: 1},  # Mais peso pra churn=0
    random_state=42
)
rf.fit(X_train_u, y_train_u)

# Teste no undersampling
y_pred_u = rf.predict(X_test_u)
print("\n📊 Resultados - RF Refinado (Undersampling, ~1761 linhas)")
print(classification_report(y_test_u, y_pred_u))

# Teste no original
y_pred_orig = rf.predict(X_test_orig)
print("\n📊 Resultados - RF Refinado (Original, ~20k linhas)")
print(classification_report(y_test_orig, y_pred_orig))

# Ajuste de threshold
y_probs_orig = rf.predict_proba(X_test_orig)[:, 1]
thresholds = [0.65, 0.7, 0.75, 0.8]
for thresh in thresholds:
    y_pred_adjusted = (y_probs_orig > thresh).astype(int)
    print(f"\n📊 Resultados - RF Refinado com Threshold {thresh} (Original)")
    print(classification_report(y_test_orig, y_pred_adjusted))

# Feature importance
importances = pd.Series(rf.feature_importances_, index=X_train_u.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
importances.plot(kind='bar', title='Feature Importance - RF Refinado')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Confusion matrix (threshold 0,75, por exemplo)
y_pred_adjusted = (y_probs_orig > 0.75).astype(int)
cm = confusion_matrix(y_test_orig, y_pred_adjusted)
ConfusionMatrixDisplay(cm, display_labels=['No Churn', 'Churn']).plot()
plt.title('Confusion Matrix - RF Refinado (Threshold 0.75)')
plt.savefig('confusion_matrix.png')
plt.close()

# Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x='churn', y='logins_per_month', data=df)
plt.title('Logins por Churn')
plt.savefig('boxplot_logins.png')
plt.close()
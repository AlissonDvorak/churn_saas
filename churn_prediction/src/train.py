

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, make_scorer, f1_score

def train_churn_pipeline(data_path, output_dir="models", figure_dir="figures", random_state=42):
    """
    Treina um pipeline de churn com Random Forest, faz undersampling, tunagem, avaliação,
    gera gráficos e salva o modelo pra API.
    
    Args:
        data_path (str): Caminho do dataset (.pkl).
        output_dir (str): Diretório pra salvar o modelo.
        figure_dir (str): Diretório pra salvar gráficos.
        random_state (int): Semente pra reprodutibilidade.
    
    Returns:
        dict: Resultados (modelo, métricas, thresholds).
    """
    # Carrega dados
    df = pd.read_pickle(data_path)
    df.drop(columns=['customer_id'], inplace=True)

    # One-hot encoding
    df_encoded = pd.get_dummies(df, columns=['company_size'], drop_first=True)

    # Undersampling estratificado
    df_majority = df_encoded[df_encoded['churn'] == 0]
    df_minority = df_encoded[df_encoded['churn'] == 1]
    frac = len(df_minority) / len(df_majority)
    df_majority_balanced = (
        df_majority.groupby(df['company_size'])
        .apply(lambda x: x.sample(frac=frac, random_state=random_state))
        .reset_index(drop=True)
    )
    df_under = pd.concat([df_majority_balanced, df_minority], axis=0).sample(frac=1, random_state=random_state)

    # Dados undersampling
    X_under = df_under.drop(columns=['churn'])
    y_under = df_under['churn']
    X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(
        X_under, y_under, test_size=0.2, random_state=random_state
    )

    # Dados originais
    X_orig = df_encoded.drop(columns=['churn'])
    y_orig = df_encoded['churn']
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
        X_orig, y_orig, test_size=0.2, random_state=random_state
    )

    # Verifica colunas
    assert list(X_train_u.columns) == list(X_test_orig.columns), "Colunas de treino e teste não batem!"

    # Tunagem
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [10, 15, 20, 25, 30, 35],
        'min_samples_split': [5, 10, 15, 20, 25],
        'min_samples_leaf': [2, 4, 6, 8, 10],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False],
        'class_weight': ['balanced', None]
    }
    rf = RandomForestClassifier(random_state=random_state)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        rf,
        param_distributions=param_grid,
        n_iter=50,
        cv=cv,
        scoring=make_scorer(f1_score),
        n_jobs=-1,
        random_state=random_state
    )
    search.fit(X_train_u, y_train_u)
    best_rf = search.best_estimator_

    # Avaliação undersampling
    y_pred_u = best_rf.predict(X_test_u)
    report_under = classification_report(y_test_u, y_pred_u, output_dict=True)
    print("\n📊 Resultados - RF Tunado (Undersampling, ~1761 linhas)")
    print(classification_report(y_test_u, y_pred_u))

    # Avaliação original
    y_pred_orig = best_rf.predict(X_test_orig)
    report_orig = classification_report(y_test_orig, y_pred_orig, output_dict=True)
    print("\n📊 Resultados - RF Tunado (Original, ~20k linhas)")
    print(classification_report(y_test_orig, y_pred_orig))

    # Thresholds
    y_probs_orig = best_rf.predict_proba(X_test_orig)[:, 1]
    threshold_reports = {}
    thresholds = [0.6, 0.65, 0.7, 0.75, 0.8]
    for thresh in thresholds:
        y_pred_adjusted = (y_probs_orig > thresh).astype(int)
        report = classification_report(y_test_orig, y_pred_adjusted, output_dict=True)
        threshold_reports[thresh] = report
        print(f"\n📊 Resultados - RF Tunado com Threshold {thresh} (Original)")
        print(classification_report(y_test_orig, y_pred_adjusted))

    # Gráficos
    # Feature Importance
    importances = pd.Series(best_rf.feature_importances_, index=X_train_u.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    importances.plot(kind='bar', title='Feature Importance - RF Tunado')
    plt.tight_layout()
    plt.savefig(f"{figure_dir}/feature_importance.png")
    plt.close()

    # Confusion Matrix (threshold 0.75)
    y_pred_adjusted = (y_probs_orig > 0.75).astype(int)
    cm = confusion_matrix(y_test_orig, y_pred_adjusted)
    ConfusionMatrixDisplay(cm, display_labels=['No Churn', 'Churn']).plot()
    plt.title('Confusion Matrix - RF Tunado (Threshold 0.75)')
    plt.savefig(f"{figure_dir}/confusion_matrix.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test_orig, y_probs_orig)
    roc_auc = roc_auc_score(y_test_orig, y_probs_orig)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC - RF Tunado')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{figure_dir}/roc_curve.png")
    plt.close()

    # Boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='churn', y='logins_per_month', data=df)
    plt.title('Logins por Mês vs Churn')
    plt.savefig(f"{figure_dir}/boxplot_logins.png")
    plt.close()

    # Salva o modelo
    joblib.dump(best_rf, f"{output_dir}/churn_rf_model.pkl")

    # Retorna resultados
    return {
        'model': best_rf,
        'report_under': report_under,
        'report_orig': report_orig,
        'threshold_reports': threshold_reports,
        'feature_importance': importances,
        'roc_auc': roc_auc
    }

if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    results = train_churn_pipeline("processed/cleaned_data/cleaned_data.pkl")
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/resumo_final.txt", "w") as f:
        f.write("📊 Resumo Final:\n")
        f.write(f"Recall Undersampling: {results['report_under']['1']['recall']*100:.1f}%\n")
        f.write(f"Precisão Undersampling: {results['report_under']['1']['precision']*100:.1f}%\n")

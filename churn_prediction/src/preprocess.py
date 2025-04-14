import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def clean_churn_dataset(path, save_path=None, plot_path=None, pickle_format=False):
    """
    - Pipeline para limpar dataset de churn.
    - Trata ~7% de NaNs com mediana ou zeros (contexto SaaS).
    - Corrige ~2% de outliers com clipagem (ex.: age 18-70).
    - Padroniza company_size (~2% de variações) em 3 categorias.
    - Gera gráficos (heatmap, boxplots, contagens) pra validação.
    - Salva como CSV ou pickle.
"""
    # ===================== #
    # === CARREGAR ======== #
    df = pd.read_csv(path)

    # ===================== #
    # === TRATAMENTO ====== #
    # ===================== #
    
    df['company_size'] = (
        df['company_size']
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({
            'small': 'pequena',
            'pequena': 'pequena',
            'media': 'média',
            'média': 'média',
            'medium': 'média',
            'grande': 'grande',
            'large': 'grande'
        })
    )

    df['age'] = df['age'].fillna(df['age'].median()).clip(18, 70)
    df['logins_per_month'] = df['logins_per_month'].fillna(df['logins_per_month'].median()).clip(0, 100)
    df['plan_value'] = df['plan_value'].fillna(df['plan_value'].median()).clip(50, 500)
    df['avg_response_time'] = df['avg_response_time'].fillna(df['avg_response_time'].median()).clip(1, 48)
    df['support_tickets'] = df['support_tickets'].fillna(0)
    df['avg_session_time'] = df.groupby('company_size')['avg_session_time'].transform(lambda x: x.fillna(x.median()))

    # ===================== #
    # ===== VISUALIZAÇÕES = #
    # ===================== #

    if plot_path:
        os.makedirs(plot_path, exist_ok=True)

        # 1. Mapa de NaNs
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isna(), cbar=False, cmap="viridis")
        plt.title("Mapa de valores ausentes")
        plt.savefig(os.path.join(plot_path, "missing_values_heatmap.png"))
        plt.close()

        # 2. Boxplots de variáveis numéricas
        num_cols = ["age", "logins_per_month", "avg_session_time", "plan_value", "avg_response_time"]
        plt.figure(figsize=(12, 8))
        for i, col in enumerate(num_cols, 1):
            plt.subplot(2, 3, i)
            sns.boxplot(y=df[col])
            plt.title(col)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, "boxplots_numericos.png"))
        plt.close()

        # 3. Distribuição de company_size
        plt.figure(figsize=(8, 5))
        sns.countplot(x="company_size", data=df)
        plt.title("Distribuição de company_size")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, "distribuicao_company_size.png"))
        plt.close()

        # 4. Distribuição de churn
        plt.figure(figsize=(6, 4))
        sns.countplot(x="churn", data=df)
        plt.title("Distribuição de churn")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, "distribuicao_churn.png"))
        plt.close()

    # ===================== #
    # ===== SALVAR ========= #
    # ===================== #
    if save_path:
        save_dir = os.path.dirname(save_path)  # Diretório do arquivo
        os.makedirs(save_dir, exist_ok=True)   # Cria o diretório se não existir
        
        if pickle_format:
            df.to_pickle(save_path)
            print(f"Dataset limpo salvo como pickle em: {save_path}")
        else:
            df.to_csv(save_path, index=False)
            print(f"Dataset limpo salvo como CSV em: {save_path}")

    return df


# Exemplo de uso
df = clean_churn_dataset(
    path="churn_prediction/data/synthetic_churn_data.csv",
    save_path="processed/cleaned_data/cleaned_data.pkl",
    plot_path="artefatos/plots/",
    pickle_format=True
)

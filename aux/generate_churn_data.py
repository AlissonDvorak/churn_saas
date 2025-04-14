import pandas as pd
import numpy as np
import random
from faker import Faker

# Configurar seed para reprodutibilidade
np.random.seed(42)
random.seed(42)

# Inicializar Faker para IDs realistas
fake = Faker()

# Parâmetros
n_rows = 100_000
churn_rate = 0.15  # 15% de churn
missing_rate = 0.07  # 7% de valores ausentes
outlier_rate = 0.02  # 2% de outliers

# Função para gerar company_size com inconsistências
def generate_company_size():
    sizes = ["pequena", "média", "grande"]
    if random.random() < 0.05:  # 5% de chance de inconsistência
        choice = random.choice(sizes)
        if choice == "pequena":
            return random.choice(["Pequena", "pequena ", "SMALL", "PequenA"])
        elif choice == "média":
            return random.choice(["Média", "media", "MEDIUM", "média "])
        else:
            return random.choice(["Grande", "grande ", "LARGE", "GrAnDe"])
    return random.choice(sizes)

# Função para gerar dados com ruído
def generate_noisy_data():
    data = {
        "customer_id": [f"CUST{str(i).zfill(5)}" for i in range(1, n_rows + 1)],
        "age": np.random.normal(40, 10, n_rows).astype(int),
        "company_size": [generate_company_size() for _ in range(n_rows)],
        "logins_per_month": np.random.poisson(20, n_rows),
        "avg_session_time": np.random.normal(30, 10, n_rows),
        "features_used": np.random.uniform(10, 90, n_rows),
        "plan_value": np.random.normal(200, 50, n_rows),
        "payment_delays": np.random.exponential(5, n_rows).astype(int),
        "support_tickets": np.random.poisson(2, n_rows),
        "avg_response_time": np.random.normal(12, 5, n_rows),
    }

    # Gerar churn com dependência de features
    churn_prob = (
        0.3 * (data["logins_per_month"] < 10) +
        0.3 * (data["payment_delays"] > 10) +
        0.2 * (data["features_used"] < 30) +
        0.2 * (data["support_tickets"] > 5)
    )
    churn_prob = np.clip(churn_prob, 0, 1) * churn_rate / 0.3  # Normalizar
    data["churn"] = np.random.binomial(1, churn_prob)

    return pd.DataFrame(data)

# Adicionar ruídos
def add_noise(df):
    # Valores ausentes
    for col in ["avg_session_time", "support_tickets", "age", "avg_response_time"]:
        mask = np.random.random(n_rows) < missing_rate
        df.loc[mask, col] = np.nan

    # Outliers
    outlier_mask = np.random.random(n_rows) < outlier_rate
    df.loc[outlier_mask, "logins_per_month"] = np.random.randint(200, 500, size=outlier_mask.sum())
    df.loc[outlier_mask, "plan_value"] = np.random.choice([-100, 1000], size=outlier_mask.sum())
    df.loc[outlier_mask, "age"] = np.random.choice([-5, 150], size=outlier_mask.sum())
    df.loc[outlier_mask, "avg_response_time"] = np.random.choice([0, 1000], size=outlier_mask.sum())

    # Ruído aleatório em variáveis numéricas
    noise = np.random.normal(0, 1, n_rows)
    df["avg_session_time"] = df["avg_session_time"] + noise
    df["plan_value"] = df["plan_value"] + noise * 5

    return df

# Gerar e salvar dataset
def main():
    df = generate_noisy_data()
    df = add_noise(df)
    
    # Garantir tipos corretos
    df["age"] = df["age"].astype(float)  # Permitir NaN
    df["support_tickets"] = df["support_tickets"].astype(float)  # Permitir NaN
    df["churn"] = df["churn"].astype(int)
    
    # Salvar CSV
    df.to_csv("synthetic_churn_data.csv", index=False)
    print(f"Dataset gerado com {len(df)} linhas e salvo como 'synthetic_churn_data.csv'")

if __name__ == "__main__":
    main()
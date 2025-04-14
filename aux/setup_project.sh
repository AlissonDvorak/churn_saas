#!/bin/bash

# Nome do projeto
PROJECT_DIR="churn_prediction"

# Criar diretório principal
mkdir -p $PROJECT_DIR

# Criar subdiretórios
mkdir -p $PROJECT_DIR/data
mkdir -p $PROJECT_DIR/models
mkdir -p $PROJECT_DIR/notebooks
mkdir -p $PROJECT_DIR/src

# Criar arquivos .py em src/ com comentário básico
touch $PROJECT_DIR/src/preprocess.py
echo "# Script para pré-processamento de dados" > $PROJECT_DIR/src/preprocess.py

touch $PROJECT_DIR/src/train.py
echo "# Script para treinamento do modelo" > $PROJECT_DIR/src/train.py

touch $PROJECT_DIR/src/predict.py
echo "# Script para predições" > $PROJECT_DIR/src/predict.py

touch $PROJECT_DIR/src/dashboard.py
echo "# Script para dashboard com Streamlit" > $PROJECT_DIR/src/dashboard.py

# Criar outros arquivos
touch $PROJECT_DIR/requirements.txt
touch $PROJECT_DIR/README.md
touch $PROJECT_DIR/LICENSE

echo "Estrutura criada com sucesso em $PROJECT_DIR!"
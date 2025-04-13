# Previsão de Churn para a CloudSync

Bem-vindo ao repositório do meu projeto de previsão de churn, uma solução de machine learning desenvolvida para identificar clientes em risco de cancelar assinaturas em uma plataforma SaaS **fictícia**, a CloudSync. Este projeto combina técnicas de classificação, pipelines escaláveis e visualizações práticas para entregar insights acionáveis, mostrando como dados podem impulsionar decisões de negócio.

---

## Contexto do Projeto

### A História por Trás do Projeto

Em 2024, fui desafiado a resolver um problema crítico para a **CloudSync**, uma startup fictícia de SaaS que oferece uma plataforma de colaboração em tempo real para equipes remotas. Com milhares de clientes usando a ferramenta para gerenciar projetos e integrar fluxos de trabalho, a empresa estava crescendo rápido — mas também enfrentando um obstáculo: a taxa de *churn* (cancelamento de assinaturas) estava subindo. O time de Customer Success notava sinais preocupantes, como clientes reduzindo o uso da plataforma ou abrindo tickets de suporte com frequência, mas não havia clareza sobre quem estava realmente em risco.

A diretoria da CloudSync queria uma solução proativa: **um modelo de machine learning que previsse quais clientes tinham maior chance de churn**, permitindo ações como suporte dedicado, descontos personalizados ou campanhas de reengajamento. Mais do que prever, a solução precisava explicar *por que* um cliente estava em risco (ex.: baixa atividade ou atrasos em pagamentos) e ser apresentada de forma acessível, como um dashboard interativo, para que gestores tomassem decisões rápidas.

### O Desafio

Prever churn é um problema clássico em SaaS, mas a CloudSync trouxe complexidades reais. O dataset continha ruídos — valores ausentes, inconsistências (como tamanhos de empresa registrados de formas diferentes) e outliers (ex.: valores de plano irreais). Com apenas ~15% dos clientes churnando, o desbalanceamento exigia cuidado para não ignorar casos raros, mas críticos. Além disso, a empresa já tinha 100 mil clientes e precisava de uma solução escalável que processasse grandes volumes de dados sem travar.

O projeto não podia ser apenas um modelo teórico. A CloudSync precisava de:
- Um **pipeline robusto** que limpasse dados sujos e escalasse para milhares de registros.
- **Previsões confiáveis**, com foco em identificar a maioria dos churns (alto *recall*), mesmo que isso gerasse alguns falsos positivos.
- **Explicações acionáveis**, para que o time de vendas soubesse exatamente onde intervir.
- Uma **interface visual** que destacasse clientes em risco e seus principais fatores de churn.

### Por Que Esse Projeto Importa

Reduzir o churn é proteger a receita recorrente, o coração de qualquer SaaS. Na CloudSync, estimava-se que cada 1% de redução no churn poderia aumentar a receita anual em dezenas de milhares de dólares. Este projeto vai além de um exercício técnico: ele mostra como machine learning pode transformar dados em estratégias de retenção, equilibrando precisão técnica com impacto de negócio.

---

## Requisitos do Projeto

Para garantir que a solução atendesse às necessidades da CloudSync, defini os seguintes requisitos:

### Requisitos Funcionais
- **RF1**: Carregar e processar um dataset com dados de clientes, incluindo uso da plataforma, interações com suporte, planos contratados e demografia.
- **RF2**: Treinar um modelo de classificação binária (*churn* = sim/não) com foco em identificar clientes em risco.
- **RF3**: Gerar probabilidades de churn para cada cliente, permitindo priorizar ações.
- **RF4**: Explicar as predições, destacando fatores que mais contribuem para o risco de churn (ex.: poucos logins ou muitos tickets).
- **RF5**: Apresentar resultados em um dashboard interativo, com tabelas e gráficos para clientes em risco.
- **RF6**: Produzir um relatório com métricas de desempenho e recomendações de retenção baseadas nas previsões.

### Requisitos Não Funcionais
- **RNF1 - Desempenho**: O modelo deve alcançar *recall* de pelo menos 80% para capturar a maioria dos churns.
- **RNF2 - Escalabilidade**: O pipeline deve processar datasets de até 100 mil linhas em menos de 5 minutos em um laptop padrão.
- **RNF3 - Interpretabilidade**: Usar ferramentas como SHAP para explicar predições, garantindo transparência.
- **RNF4 - Reprodutibilidade**: O código deve ser versionado no GitHub, com instruções claras para execução.
- **RNF5 - Usabilidade**: O dashboard deve ser intuitivo, com filtros para explorar clientes por nível de risco.
- **RNF6 - Manutenibilidade**: O código deve ser modular, com pastas organizadas e comentários explicativos.

### Requisitos Técnicos
- **Linguagem**: Python 3.8+.
- **Bibliotecas**:
  - Dados: Pandas, NumPy.
  - Modelagem: Scikit-learn (Random Forest, Logistic Regression), XGBoost.
  - Explicabilidade: SHAP.
  - Visualização: Matplotlib, Seaborn, Plotly.
  - Interface: Streamlit.
- **Ambiente**: Virtualenv ou Conda, com dependências listadas em `requirements.txt`.
- **Ferramentas**:
  - Git para versionamento.
  - Jupyter Notebooks para exploração inicial, mas scripts `.py` para o pipeline final.
- **Dataset**: Um dataset sintético com 100 mil linhas, contendo features como:
  - Demográficas: `age`, `company_size`.
  - Uso: `logins_per_month`, `avg_session_time`, `features_used`.
  - Financeiras: `plan_value`, `payment_delays`.
  - Suporte: `support_tickets`, `avg_response_time`.
  - Target: `churn` (0 ou 1, ~15% de churns).
  - Inclui ruídos: valores ausentes (~7%), outliers (ex.: idades negativas), inconsistências (ex.: strings mal formatadas).

---

## O Que Fazer: Passos para Executar o Projeto

Para transformar esse desafio em uma solução funcional, aqui está o plano de ação detalhado. Se você é um recrutador, isso mostra minha abordagem estruturada; se está replicando o projeto, siga os passos abaixo!

### 1. Configurar o Ambiente
- **Tarefa**: Criar um ambiente virtual e instalar dependências.
- **Como fazer**:
  ```bash
  python -m venv venv
  source venv/bin/activate  # Linux/Mac
  venv\Scripts\activate     # Windows
  pip install -r requirements.txt
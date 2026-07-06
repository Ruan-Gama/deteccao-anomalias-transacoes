# Detecção de Anomalias em Transações Financeiras

Projeto de detecção de fraudes em transações de cartão de crédito utilizando **Isolation Forest**, um algoritmo de aprendizado não supervisionado para detecção de anomalias.

## Contexto

Fraudes em transações financeiras são raras em relação ao volume total de transações legítimas, o que torna a classificação supervisionada tradicional difícil de aplicar sem técnicas específicas para dados desbalanceados. Este projeto trata o problema como uma tarefa de detecção de anomalias: o modelo aprende o padrão das transações normais e sinaliza como anômalo tudo o que foge desse padrão, sem depender dos rótulos de fraude durante o treinamento.

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud), disponibilizado pela ULB (Université Libre de Bruxelles).

- 284.807 transações realizadas por portadores de cartão europeus em setembro de 2013
- 31 atributos: `Time`, `Amount`, `V1` a `V28` (componentes de uma transformação PCA aplicada por questões de confidencialidade) e `Class` (0 = normal, 1 = fraude)
- Altamente desbalanceado: apenas ~0,17% das transações são fraudulentas

## Metodologia

1. **Análise exploratória**: distribuição dos valores das transações, desbalanceamento entre classes, padrão de transações por horário do dia e correlação de cada variável com a ocorrência de fraude.
2. **Pré-processamento**: padronização (`StandardScaler`) aplicada apenas às colunas `Time` e `Amount`, já que as componentes da PCA já estão em escala comparável entre si.
3. **Separação treino/teste**: divisão estratificada para avaliar o modelo em dados que ele nunca viu durante o treinamento.
4. **Treinamento**: `IsolationForest` treinado apenas com dados de treino, sem uso da variável alvo.
5. **Avaliação**: matriz de confusão, classification report, ROC-AUC e Average Precision (mais adequadas que acurácia para problemas desbalanceados), além da distribuição do score de anomalia por classe.
6. **Persistência**: modelo e scaler salvos com `joblib` para reuso posterior.

## Estrutura do repositório

```
.
├── deteccao_anomalias_transacoes.ipynb   # notebook principal
├── requirements.txt                       # dependências do projeto
└── README.md
```

## Como executar

O notebook foi desenvolvido para rodar no Google Colab (usa `kagglehub` para baixar o dataset diretamente do Kaggle). Para executar localmente:

```bash
pip install -r requirements.txt
```

É necessário ter uma conta no Kaggle e as credenciais configuradas (`kaggle.json`) para o `kagglehub` baixar o dataset automaticamente. Como alternativa, o dataset pode ser baixado manualmente em [kaggle.com/mlg-ulb/creditcardfraud](https://www.kaggle.com/mlg-ulb/creditcardfraud) e carregado localmente com `pd.read_csv("creditcard.csv")`.

## Resultados

O modelo consegue identificar parte significativa das transações fraudulentas, mas ainda apresenta falsos positivos e fraudes não detectadas, o que é esperado em uma abordagem puramente não supervisionada. As métricas ROC-AUC e Average Precision, calculadas sobre o conjunto de teste, indicam a qualidade da ordenação das transações por risco produzida pelo modelo.

## Possíveis melhorias

- Calibrar o limiar de decisão a partir da curva precision-recall em vez de assumir a taxa real de fraude como `contamination`
- Comparar com outros algoritmos de detecção de anomalias (Local Outlier Factor, One-Class SVM, Autoencoders)
- Engenharia de atributos adicional a partir de `Time` (padrões por dia da semana, janelas móveis)
- Ajuste fino de hiperparâmetros via validação cruzada

## Tecnologias

- Python
- pandas / numpy
- scikit-learn
- matplotlib / seaborn
- kagglehub
- joblib

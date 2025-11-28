README – Classificação de Recorrência de Câncer de Mama com Decision Tree
Descrição do Projeto
Este projeto treina um modelo de Machine Learning utilizando árvore de decisão (DecisionTreeClassifier) 
para prever se um caso de câncer de mama terá recorrência ou não recorrência, com base em atributos clínicos.

O sistema realiza:
Pré-processamento dos dados
Treinamento do modelo
Cálculo da acurácia
Exibição da matriz de confusão
Classificação de novos exemplos

Passos do Código
1. Carregamento dos dados
O arquivo CSV é lido e linhas com dados faltantes são removidas.

2. Pré-processamento
Como o dataset contém variáveis categóricas, elas são convertidas em variáveis numéricas usando One-Hot Encoding (pd.get_dummies).

3. Divisão Treino/Teste
Os dados são divididos em:
70% treino
30% teste

4. Treinamento do Modelo
Um classificador DecisionTreeClassifier é treinado com os dados processados.

5. Avaliação
O modelo retorna:
Acurácia
Matriz de Confusão
6. Classificação de Novas Instâncias

O código contém um exemplo de entrada de um novo paciente que é classificado pelo modelo já treinado.

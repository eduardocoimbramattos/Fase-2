# Relatorio tecnico - Tech Challenge B

## Problema escolhido

O dataset analisado e o arquivo anexado `data.csv`, localizado em `data.csv`.
A tarefa foi formulada como um problema de classificacao binaria: prever se um
caso deve ser marcado como `Dangerous = Yes` ou `Dangerous = No` a partir do
animal e de cinco sintomas observados.

Embora o enunciado use exemplos de diagnostico humano, este dataset representa
um cenario clinico/veterinario de triagem. A solucao deve ser interpretada como
apoio inicial a decisao, nunca como diagnostico final automatico.

## Exploracao dos dados

- Linhas originais: 871
- Linhas apos limpeza: 841
- Duplicatas removidas: 28
- Colunas: AnimalName, symptoms1, symptoms2, symptoms3, symptoms4, symptoms5, Dangerous
- Distribuicao do alvo: {'yes': 821, 'no': 20}

Foram gerados graficos de distribuicao do alvo, animais mais frequentes,
sintomas mais frequentes, features numericas derivadas e valores ausentes.

## Pre-processamento

As estrategias utilizadas foram:

- padronizacao de texto com `strip`, caixa baixa e normalizacao de espacos;
- correcao de algumas inconsistencias textuais evidentes, como `seizuers` para `seizures`;
- remocao de duplicatas;
- mapeamento de `Dangerous` para alvo binario;
- preenchimento de sintomas ausentes com `unknown`;
- criacao de features numericas derivadas: quantidade de sintomas unicos,
  quantidade de sintomas desconhecidos e tamanho textual dos sintomas;
- `OneHotEncoder` para variaveis categoricas;
- `StandardScaler` para features numericas;
- pipeline do scikit-learn para evitar vazamento de dados entre treino, validacao e teste.

## Correlacao

A correlacao foi calculada de duas formas:

- correlacao de Pearson entre dummies one-hot e o alvo binario;
- Cramer's V entre cada variavel categorica original e o alvo.

Essas tabelas foram salvas em `tables/encoded_feature_correlations.csv` e
`tables/categorical_association_cramers_v.csv`.

## Modelagem

Foram treinados quatro modelos, cobrindo tecnicas lineares, arvores e metodos
baseados em vizinhanca:

- Regressao Logistica;
- Arvore de Decisao;
- Random Forest;
- KNN.

A separacao foi feita em treino, validacao e teste. O conjunto de validacao foi
usado para escolher o melhor modelo, priorizando `recall` da classe `Yes`, pois
em triagem clinica e mais grave deixar de sinalizar um caso perigoso do que
gerar um alerta falso. O F1-score foi usado como criterio de equilibrio.

### Metricas de validacao

| model               | accuracy | precision_yes | recall_yes | f1_yes | f1_weighted | roc_auc |
| ------------------- | -------- | ------------- | ---------- | ------ | ----------- | ------- |
| knn                 | 0.9762   | 0.9762        | 1.0000     | 0.9880 | 0.9644      | 0.8369  |
| random_forest       | 0.9821   | 0.9879        | 0.9939     | 0.9909 | 0.9809      | 0.9268  |
| logistic_regression | 0.9702   | 0.9818        | 0.9878     | 0.9848 | 0.9682      | 0.9070  |
| decision_tree       | 0.9107   | 0.9934        | 0.9146     | 0.9524 | 0.9365      | 0.8567  |

Modelo selecionado: `knn`.

### Metricas do modelo selecionado

| dataset    | accuracy | precision_yes | recall_yes | f1_yes | f1_weighted | roc_auc |
| ---------- | -------- | ------------- | ---------- | ------ | ----------- | ------- |
| train      | 1.0000   | 1.0000        | 1.0000     | 1.0000 | 1.0000      | 1.0000  |
| validation | 0.9762   | 0.9762        | 1.0000     | 0.9880 | 0.9644      | 0.8369  |
| test       | 0.9763   | 0.9763        | 1.0000     | 0.9880 | 0.9646      | 0.9833  |

## Interpretacao

O script gera importancia de variaveis nativa quando o modelo permite
(`feature_importances_` ou coeficientes). Quando isso nao esta disponivel,
usa permutation importance. SHAP e executado automaticamente quando a biblioteca
esta instalada e compativel com o modelo selecionado.

Status SHAP: SHAP estava instalado, mas falhou neste modelo/ambiente. Fallback mantido com feature importance. Erro: The passed model is not callable and cannot be analyzed directly with the given masker! Model: KNeighborsClassifier(n_neighbors=7, weights='distance')

Top features:

| feature               | importance |
| --------------------- | ---------- |
| symptoms1             | 0.0020     |
| unique_symptom_count  | 0.0000     |
| symptoms2             | 0.0000     |
| symptoms3             | 0.0000     |
| symptoms4             | 0.0000     |
| symptom_text_length   | 0.0000     |
| unknown_symptom_count | 0.0000     |
| AnimalName            | -0.0002    |
| symptoms5             | -0.0005    |

## Analise critica

O modelo pode ser util como ferramenta de apoio a triagem, destacando casos que
merecem revisao prioritaria. Entretanto, seu uso pratico exige cuidado:

- o dataset e pequeno e limitado ao universo de animais/sintomas observados;
- categorias novas podem aparecer em producao;
- correlacao nao implica causalidade;
- sintomas textuais foram tratados como categorias, sem contexto clinico amplo;
- e necessario validar o modelo com dados externos e acompanhamento de
  profissionais antes de qualquer uso real.

Assim, a recomendacao e usar a solucao como alerta inicial e ferramenta de
organizacao do fluxo de atendimento. A decisao final deve permanecer com o(a)
medico(a) ou profissional responsavel.

## Artefatos gerados

- Modelo: `models/best_model.pkl`
- Metricas: `tables/validation_metrics.csv` e `tables/best_model_metrics.csv`
- Relatorios de classificacao: `tables/classification_reports.json`
- Exemplo de predicao: `tables/prediction_example.json`
- README: `outputs_techchallenge_b\docs\README.md`
- Dockerfile: `outputs_techchallenge_b\docs\Dockerfile`
- Roteiro do video: `outputs_techchallenge_b\docs\roteiro_video_demo.md`

## Figuras

- `outputs_techchallenge_b\figures\01_target_distribution.png`
- `outputs_techchallenge_b\figures\02_top_animals.png`
- `outputs_techchallenge_b\figures\03_top_symptoms.png`
- `outputs_techchallenge_b\figures\04_numeric_features_by_target.png`
- `outputs_techchallenge_b\figures\05_missing_values.png`
- `outputs_techchallenge_b\figures\06_top_correlations.png`
- `outputs_techchallenge_b\figures\07_cramers_v.png`
- `outputs_techchallenge_b\figures\08_confusion_matrix_train.png`
- `outputs_techchallenge_b\figures\08_confusion_matrix_validation.png`
- `outputs_techchallenge_b\figures\08_confusion_matrix_test.png`
- `outputs_techchallenge_b\figures\09_feature_importance.png`

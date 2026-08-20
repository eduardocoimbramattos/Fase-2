# Relatorio tecnico: otimizacao de modelos tabulares por algoritmos geneticos

## 1. Resumo

Este trabalho apresenta a aplicacao de algoritmos geneticos a quatro modelos de classificacao para identificar registros potencialmente perigosos a partir do animal e de cinco sintomas: Regressao Logistica, Arvore de Decisao, Random Forest e KNN.

A busca genetica foi realizada separadamente para cada modelo, com espacos de hiperparametros especificos. Foram executados tres experimentos com diferentes tamanhos de populacao, numero de geracoes e taxas de mutacao. A funcao de fitness combinou accuracy, recall e F1-score, atribuindo maior peso ao recall da classe perigosa.

O melhor resultado operacional foi obtido pelo Random Forest otimizado no experimento E2, com fitness de validacao igual a 0.9919 e accuracy de teste igual a 0.9941.

## 2. Dados e pre-processamento

O dataset utilizado foi `data.csv`, contendo `AnimalName`, cinco colunas de sintomas e `Dangerous` como variavel-alvo.

O pre-processamento incluiu normalizacao de textos, remocao de duplicatas, preenchimento de sintomas ausentes com `unknown`, conversao do alvo para formato binario, criacao das features `unique_symptom_count`, `unknown_symptom_count` e `symptom_text_length`, codificacao categorica com `OneHotEncoder` e padronizacao numerica com `StandardScaler`.

Os dados foram separados em treino, validacao e teste com estratificacao. O conjunto de teste permaneceu isolado durante a busca genetica.

## 3. Modelos avaliados

Foram utilizados quatro modelos: Regressao Logistica, Arvore de Decisao, Random Forest e KNN. Os modelos originais utilizaram hiperparametros fixos. Na etapa otimizada, cada modelo recebeu uma codificacao genetica propria.

## 4. Codificacao dos genes

Cada individuo representa uma configuracao de hiperparametros:

| Modelo | Hiperparametros codificados |
|---|---|
| Regressao Logistica | `C`, `penalty`, `class_weight` |
| Arvore de Decisao | `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `class_weight` |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `class_weight` |
| KNN | `n_neighbors`, `weights`, `p`, `leaf_size` |

Genes inteiros foram sorteados em intervalos definidos. Genes categoricos foram escolhidos a partir de listas de valores validos para cada estimador.

## 5. Funcao de fitness

Cada individuo foi avaliado por validacao cruzada estratificada com cinco particionamentos. A funcao de fitness foi:

```python
fitness = (
    0.30 * accuracy_cv
    + 0.40 * recall_cv
    + 0.30 * f1_cv
)
```

O recall recebeu o maior peso porque, em triagem, deixar de identificar um caso perigoso pode ser mais grave do que produzir um alerta falso. Accuracy e F1 foram incluidos para considerar o desempenho geral e o equilibrio entre precision e recall. O conjunto de teste nao participou do fitness.

## 6. Operadores geneticos

A populacao inicial foi formada por individuos aleatorios dentro dos espacos validos. Os individuos foram ordenados pelo fitness e os melhores foram preservados por elitismo. Os pais foram selecionados entre os melhores individuos.

O cruzamento foi realizado gene a gene: cada gene do filho foi herdado do primeiro ou do segundo pai com probabilidade de 50%. Na mutacao, um gene foi escolhido aleatoriamente e substituido por outro valor valido. A taxa de mutacao variou entre os experimentos.

## 7. Experimentos realizados

Foram realizados tres experimentos para cada modelo, totalizando doze buscas geneticas:

| Experimento | Populacao | Geracoes | Taxa de mutacao |
|---|---:|---:|---:|
| E1_pop6_mut10 | 6 | 4 | 10% |
| E2_pop8_mut30 | 8 | 5 | 30% |
| E3_pop12_mut50 | 12 | 6 | 50% |

As configuracoes representam uma busca compacta, uma busca intermediaria e uma busca com maior diversidade genetica.

## 8. Melhores configuracoes

| Modelo | Experimento | Hiperparametros selecionados |
|---|---|---|
| Regressao Logistica | E3_pop12_mut50 | `C=10.0`, `penalty=l2`, `class_weight=None` |
| Arvore de Decisao | E1_pop6_mut10 | `max_depth=3`, `min_samples_split=14`, `min_samples_leaf=5`, `max_features=log2`, `class_weight=None` |
| Random Forest | E2_pop8_mut30 | `n_estimators=575`, `max_depth=14`, `min_samples_split=3`, `min_samples_leaf=1`, `max_features=log2`, `class_weight=balanced` |
| KNN | E3_pop12_mut50 | `n_neighbors=4`, `weights=distance`, `p=1`, `leaf_size=17` |

Os maiores fitnesses de validacao foram: Regressao Logistica 0.9919, Arvore de Decisao 0.9892, Random Forest 0.9919 e KNN 0.9919.

## 9. Comparacao entre modelos originais e otimizados

| Versao | Modelo | Conjunto | Accuracy | Recall | F1 | ROC AUC |
|---|---|---|---:|---:|---:|---:|
| Original | Regressao Logistica | Validacao | 0.9702 | 0.9878 | 0.9848 | 0.9070 |
| Original | Regressao Logistica | Teste | 0.9941 | 1.0000 | 0.9970 | 0.9591 |
| Original | Arvore de Decisao | Validacao | 0.9107 | 0.9146 | 0.9524 | 0.8567 |
| Original | Arvore de Decisao | Teste | 0.9704 | 0.9758 | 0.9847 | 0.8614 |
| Original | Random Forest | Validacao | 0.9821 | 0.9939 | 0.9909 | 0.9268 |
| Original | Random Forest | Teste | 0.9822 | 0.9879 | 0.9909 | 0.9697 |
| Original | KNN | Validacao | 0.9762 | 1.0000 | 0.9880 | 0.8369 |
| Original | KNN | Teste | 0.9763 | 1.0000 | 0.9880 | 0.9833 |
| Genetico | Regressao Logistica | Validacao | 0.9821 | 1.0000 | 0.9909 | 0.8918 |
| Genetico | Regressao Logistica | Teste | 0.9941 | 1.0000 | 0.9970 | 0.9530 |
| Genetico | Arvore de Decisao | Validacao | 0.9762 | 1.0000 | 0.9880 | 0.5000 |
| Genetico | Arvore de Decisao | Teste | 0.9763 | 1.0000 | 0.9880 | 0.5000 |
| Genetico | Random Forest | Validacao | 0.9821 | 1.0000 | 0.9909 | 0.8369 |
| Genetico | Random Forest | Teste | 0.9941 | 1.0000 | 0.9970 | 0.9727 |
| Genetico | KNN | Validacao | 0.9821 | 1.0000 | 0.9909 | 0.7241 |
| Genetico | KNN | Teste | 0.9822 | 0.9879 | 0.9909 | 0.8667 |

## 10. Analise dos resultados

A busca genetica elevou o recall de validacao dos modelos selecionados para 1.0000. O maior fitness de validacao foi 0.9919, observado na Regressao Logistica, no Random Forest e no KNN.

No teste, o Random Forest otimizado apresentou accuracy de 0.9941, recall de 1.0000 e F1 de 0.9970. A Regressao Logistica otimizada tambem apresentou accuracy de 0.9941 e F1 de 0.9970. O KNN otimizado manteve desempenho semelhante ao original. A Arvore de Decisao apresentou desempenho inferior no teste apesar da melhoria observada na validacao.

A diferenca entre validacao e teste evidencia que a selecao baseada em uma unica divisao pode produzir variacao de desempenho. Os resultados devem ser interpretados em conjunto com a validacao cruzada, o teste e a estabilidade entre experimentos.

## 11. Conclusao

O algoritmo genetico foi implementado com codificacao de hiperparametros, populacao inicial, selecao, elitismo, cruzamento, mutacao e fitness multicriterio. A estrategia foi aplicada aos quatro modelos tabulares e avaliada em tres configuracoes experimentais distintas.

O Random Forest otimizado no experimento E2 apresentou o melhor resultado operacional entre as configuracoes selecionadas, com accuracy de teste de 0.9941, recall de 1.0000 e F1 de 0.9970. O resultado atende ao objetivo de investigar a otimizacao dos modelos de diagnostico por algoritmos geneticos.

O modelo deve ser utilizado como apoio a triagem. O dataset possui categorias textuais especificas e nao substitui validacao externa nem a avaliacao de um profissional responsavel.

## 12. Artefatos

Os principais artefatos produzidos foram:

- `algoritmo_genetico.py`;
- `Modelo_Tabulado.ipynb`;
- `data.csv`;
- `genetic_experiment_results.csv`;
- `genetic_experiment_history.csv`;
- `before_after_model_metrics.csv`;
- `genetic_best_params_all_experiments.json`;
- `genetic_selection.json`;
- `best_model.pkl`.

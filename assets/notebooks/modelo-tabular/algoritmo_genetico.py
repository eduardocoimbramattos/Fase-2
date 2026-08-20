"""Busca genetica de hiperparametros para os quatro modelos tabulares.

A funcao genetic_search_model recebe um pipeline de dados ja separado em treino
(validation e teste permanecem fora da busca) e retorna o melhor estimador,
os genes vencedores e o historico das geracoes.
"""

from copy import deepcopy

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate

FITNESS_WEIGHTS = {
    "accuracy": 0.30,
    "recall": 0.40,
    "f1": 0.30,
}

GENETIC_SPACES = {
    "logistic_regression": {
        "C": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        "penalty": ["l1", "l2"],
        "class_weight": ["balanced", None],
    },
    "decision_tree": {
        "max_depth": (3, 30),
        "min_samples_split": (2, 20),
        "min_samples_leaf": (1, 10),
        "max_features": ["sqrt", "log2", None],
        "class_weight": ["balanced", None],
    },
    "random_forest": {
        "n_estimators": (150, 600),
        "max_depth": (4, 26),
        "min_samples_split": (2, 16),
        "min_samples_leaf": (1, 9),
        "max_features": ["sqrt", "log2", None],
        "class_weight": ["balanced", "balanced_subsample", None],
    },
    "knn": {
        "n_neighbors": (3, 25),
        "weights": ["uniform", "distance"],
        "p": (1, 2),
        "leaf_size": (15, 60),
    },
}


def genetic_search_model(
    model_name,
    build_estimator,
    x_train,
    y_train,
    random_state=42,
    population_size=8,
    generations=5,
    mutation_rate=0.30,
):
    """Otimiza um modelo usando fitness de accuracy, recall e F1-score."""
    rng = np.random.default_rng(random_state)
    search_space = GENETIC_SPACES[model_name]
    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )
    scoring = {
        "accuracy": "accuracy",
        "recall": "recall",
        "f1": "f1",
    }

    def sample_gene(values):
        if isinstance(values, tuple):
            return int(rng.integers(values[0], values[1] + 1))
        return values[int(rng.integers(0, len(values)))]

    def create_individual():
        return {
            gene: sample_gene(values)
            for gene, values in search_space.items()
        }

    def evaluate(individual):
        estimator = build_estimator(model_name, individual, random_state)
        scores = cross_validate(
            estimator,
            x_train,
            y_train,
            scoring=scoring,
            cv=cross_validation,
            n_jobs=-1,
        )
        metrics = {
            metric: float(np.mean(scores[f"test_{metric}"]))
            for metric in scoring
        }
        metrics["fitness"] = sum(
            FITNESS_WEIGHTS[metric] * metrics[metric]
            for metric in FITNESS_WEIGHTS
        )
        return metrics

    def mutate(individual):
        child = individual.copy()
        gene = str(rng.choice(list(search_space)))
        child[gene] = sample_gene(search_space[gene])
        return child

    def crossover(first, second):
        return {
            gene: first[gene] if rng.random() < 0.5 else second[gene]
            for gene in search_space
        }

    population = [create_individual() for _ in range(population_size)]
    history = []
    best_individual = None
    best_fitness = -np.inf

    for generation in range(generations):
        scored = [
            (individual, evaluate(individual))
            for individual in population
        ]
        scored.sort(key=lambda item: item[1]["fitness"], reverse=True)
        generation_best, generation_metrics = scored[0]
        history.append(
            {
                "generation": generation + 1,
                "best_fitness": generation_metrics["fitness"],
                "accuracy_cv": generation_metrics["accuracy"],
                "recall_cv": generation_metrics["recall"],
                "f1_cv": generation_metrics["f1"],
            }
        )

        if generation_metrics["fitness"] > best_fitness:
            best_individual = deepcopy(generation_best)
            best_fitness = generation_metrics["fitness"]

        elite_count = max(2, population_size // 4)
        parents = [individual for individual, _ in scored[:elite_count]]
        next_population = [individual.copy() for individual in parents]

        while len(next_population) < population_size:
            first = parents[int(rng.integers(0, len(parents)))]
            second = parents[int(rng.integers(0, len(parents)))]
            child = crossover(first, second)
            if rng.random() < mutation_rate:
                child = mutate(child)
            next_population.append(child)

        population = next_population

    if best_individual is None:
        raise RuntimeError("A busca genetica nao encontrou uma configuracao.")

    best_model = build_estimator(model_name, best_individual, random_state)
    best_model.fit(x_train, y_train)
    return best_model, best_individual, history


EXPERIMENTS = [
    {"population_size": 6, "generations": 4, "mutation_rate": 0.10},
    {"population_size": 8, "generations": 5, "mutation_rate": 0.30},
    {"population_size": 12, "generations": 6, "mutation_rate": 0.50},
]

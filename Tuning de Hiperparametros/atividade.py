import numpy as np
import pandas as pd
from sklearn.datasets import load_wine, load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV

# Função para avaliar modelo
def avaliar_modelo(modelo, X_teste, y_teste):
    y_pred = modelo.predict(X_teste)
    return {
        "acurácia": accuracy_score(y_teste, y_pred),
        "precisão": precision_score(y_teste, y_pred, average="weighted"),
        "revocação": recall_score(y_teste, y_pred, average="weighted"),
        "f1": f1_score(y_teste, y_pred, average="weighted")
    }

# Bases de dados
bases = {
    "Vinho": load_wine(),
    "Dígitos": load_digits(),
    "Câncer de Mama": load_breast_cancer()
}

# Modelos e hiperparâmetros
modelos = {
    "Regressão Logística": (
        LogisticRegression(max_iter=5000),
        {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l2"],
            "solver": ["lbfgs", "saga"]
        }
    ),
    "SVM": (
        SVC(),
        {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"]
        }
    ),
    "Random Forest": (
        RandomForestClassifier(),
        {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5, 10]
        }
    )
}

# Loop principal
resultados = []

for nome_base, base in bases.items():
    X, y = base.data, base.target
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

    for nome_modelo, (modelo, parametros) in modelos.items():
        # Modelo sem tuning
        modelo.fit(X_treino, y_treino)
        sem_tuning = avaliar_modelo(modelo, X_teste, y_teste)

        # GridSearchCV
        grid = GridSearchCV(modelo, parametros, cv=5, n_jobs=-1)
        grid.fit(X_treino, y_treino)
        grid_res = avaliar_modelo(grid.best_estimator_, X_teste, y_teste)

        # RandomizedSearchCV
        rand = RandomizedSearchCV(modelo, parametros, cv=5, n_jobs=-1, n_iter=5, random_state=42)
        rand.fit(X_treino, y_treino)
        rand_res = avaliar_modelo(rand.best_estimator_, X_teste, y_teste)

        # BayesSearchCV
        bayes = BayesSearchCV(modelo, parametros, cv=5, n_jobs=-1, n_iter=10, random_state=42)
        bayes.fit(X_treino, y_treino)
        bayes_res = avaliar_modelo(bayes.best_estimator_, X_teste, y_teste)

        # Guardar resultados
        for metodo, metricas in {
            "Sem Tuning": sem_tuning,
            "GridSearchCV": grid_res,
            "RandomizedSearchCV": rand_res,
            "BayesSearchCV": bayes_res
        }.items():
            resultados.append({
                "Base": nome_base,
                "Modelo": nome_modelo,
                "Método": metodo,
                "Acurácia": metricas["acurácia"],
                "Precisão": metricas["precisão"],
                "Revocação": metricas["revocação"],
                "F1": metricas["f1"]
            })

# Tabela final
df_resultados = pd.DataFrame(resultados)
print(df_resultados)

input("Concluído!")
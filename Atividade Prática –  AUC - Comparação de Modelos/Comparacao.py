# %% [markdown]
# # Checklist de avaliação de modelos — California Housing (Classificação)
# 
# - Problema: Classificação binária (MedHouseVal acima vs. abaixo da mediana)
# - Modelos: Logistic Regression, Random Forest, SVM (RBF)
# - Split: Treino (70%), Validação (15%), Teste (15%)
# - Métricas: Accuracy, F1, Recall, ROC e AUC
# - K-Fold: 5 folds, com média e desvio padrão
# - Teste de generalização: perturbação leve dos dados (simular outro contexto)
# - Entregáveis: Tabela comparativa e conclusões conforme checklist

# %% [markdown]
# ## 0. Imports e setup

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, roc_auc_score, roc_curve, RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

sns.set(style="whitegrid", context="notebook")
np.random.seed(42)

# %% [markdown]
# ## 1. Carregamento da base e preparação (discretização)
# - Discretizamos o alvo `MedHouseVal` em binário: 1 se acima da mediana, 0 caso contrário.
# - Mantemos somente atributos numéricos (a base já é numérica).
# - Não usamos o conjunto de teste para qualquer ajuste de parâmetros.

# %%
data = fetch_california_housing(as_frame=True)
df = data.frame.copy()

# alvo binário: acima da mediana = 1, abaixo/igual = 0
median_val = df["MedHouseVal"].median()
df["target"] = (df["MedHouseVal"] > median_val).astype(int)

X = df.drop(columns=["MedHouseVal", "target"])
y = df["target"]

# %% [markdown]
# ## 2. Divisão correta dos dados (70/15/15)
# - Primeiro split: treino+val vs. teste (85/15)
# - Segundo split: treino vs. validação (aprox. 70/15)

# %%
# Split 1: Treino+Val (85%) vs Teste (15%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

# Split 2: Treino (70%) vs Validação (15%) — relativo ao total
# Proporção desejada: 70/15 dentro do 85% -> val_size = 15/85 ≈ 0.17647
val_size = 0.15 / 0.85
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=val_size, stratify=y_train_val, random_state=42
)

print("Tamanhos:")
print(f"Treino: {X_train.shape[0]} ({X_train.shape[0]/len(df):.2%})")
print(f"Validação: {X_val.shape[0]} ({X_val.shape[0]/len(df):.2%})")
print(f"Teste: {X_test.shape[0]} ({X_test.shape[0]/len(df):.2%})")

# %% [markdown]
# ## 3. Definição dos modelos e pipelines
# - Logistic Regression (com padronização)
# - Random Forest (sem padronização necessária)
# - SVM RBF (com padronização)
# 
# Observação importante: o conjunto de teste permanece separado e só é usado no fim.

# %%
models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200, solver="lbfgs"))
    ]),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_split=2, random_state=42, n_jobs=-1
    ),
    "SVM_RBF": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, C=1.0, gamma="scale", random_state=42))
    ]),
}

# %% [markdown]
# ## 4. Treino e avaliação em treino × validação × teste
# - Métricas: Accuracy, F1, Recall, ROC-AUC
# - Interpretação: diferença treino vs. teste (5–10% aceitável)

# %%
def eval_model(model, X_tr, y_tr, X_va, y_va, X_te, y_te, name):
    # Treina no conjunto de treino (apenas)
    model.fit(X_tr, y_tr)

    # Predições e probabilidades
    y_tr_pred = model.predict(X_tr)
    y_va_pred = model.predict(X_va)
    y_te_pred = model.predict(X_te)

    # Probabilidades para AUC/ROC
    # Para pipelines, obter o step final
    if hasattr(model, "predict_proba"):
        y_tr_proba = model.predict_proba(X_tr)[:, 1]
        y_va_proba = model.predict_proba(X_va)[:, 1]
        y_te_proba = model.predict_proba(X_te)[:, 1]
    else:
        # SVM sem probability = True não tem predict_proba
        # mas aqui definimos probability=True, então ok.
        # Por segurança:
        try:
            y_tr_proba = model.predict_proba(X_tr)[:, 1]
            y_va_proba = model.predict_proba(X_va)[:, 1]
            y_te_proba = model.predict_proba(X_te)[:, 1]
        except Exception:
            # fallback: usar decision_function e normalizar via min-max
            def df_to_proba(m, X_):
                s = m.decision_function(X_)
                s = (s - s.min()) / (s.max() - s.min() + 1e-9)
                return s
            y_tr_proba = df_to_proba(model, X_tr)
            y_va_proba = df_to_proba(model, X_va)
            y_te_proba = df_to_proba(model, X_te)

    # Métricas
    def metrics(y_true, y_pred, y_proba):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "ROC_AUC": roc_auc_score(y_true, y_proba),
        }

    train_metrics = metrics(y_tr, y_tr_pred, y_tr_proba)
    val_metrics = metrics(y_va, y_va_pred, y_va_proba)
    test_metrics = metrics(y_te, y_te_pred, y_te_proba)

    return {
        "name": name,
        "model": model,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "proba": {"train": y_tr_proba, "val": y_va_proba, "test": y_te_proba},
        "pred": {"train": y_tr_pred, "val": y_va_pred, "test": y_te_pred},
    }

results = []
for name, model in models.items():
    res = eval_model(model, X_train, y_train, X_val, y_val, X_test, y_test, name)
    results.append(res)

# %% [markdown]
# ## 5. Validação cruzada (K-Fold = 5)
# - Usamos apenas o conjunto de treino para CV (consistente com não usar teste para ajustes)
# - Reportamos média e desvio padrão para Accuracy, F1, Recall e ROC-AUC

# %%
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def cv_scores(model, X_tr, y_tr):
    # Para pipelines e modelos com predict_proba/decision_function
    scoring = ["accuracy", "f1", "recall", "roc_auc"]
    scores = {}
    for sc in scoring:
        s = cross_val_score(model, X_tr, y_tr, cv=cv, scoring=sc, n_jobs=-1)
        scores[sc] = {"mean": np.mean(s), "std": np.std(s)}
    return scores

for res in results:
    res["cv"] = cv_scores(models[res["name"]], X_train, y_train)

# %% [markdown]
# ## 6. Curva ROC e AUC (treino e teste)
# - Plotamos ROC para os três modelos no conjunto de teste
# - Interpretação AUC: 0.5 aleatório, ~0.8 bom, >0.9 excelente

# %%
plt.figure(figsize=(8,6))
for res in results:
    fpr, tpr, _ = roc_curve(y_test, res["proba"]["test"])
    auc = roc_auc_score(y_test, res["proba"]["test"])
    plt.plot(fpr, tpr, label=f"{res['name']} (AUC={auc:.3f})")

plt.plot([0,1],[0,1],"k--", label="Aleatório")
plt.xlabel("FPR (1 - Especificidade)")
plt.ylabel("TPR (Recall)")
plt.title("Curvas ROC - Conjunto de Teste")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Teste de generalização
# - Simulamos um leve deslocamento de distribuição nos atributos (ruído gaussiano fraco)
# - Avaliamos métricas no conjunto de teste perturbado
# - Observamos manutenção ou queda do desempenho

# %%
def perturb(X, noise_scale=0.05):
    Xp = X.copy()
    # Ruído proporcional à escala de cada coluna
    for c in Xp.columns:
        col = Xp[c].values
        std = np.std(col)
        noise = np.random.normal(0, noise_scale * (std + 1e-9), size=col.shape)
        Xp[c] = col + noise
    return Xp

X_test_shift = perturb(X_test, noise_scale=0.05)

for res in results:
    model = res["model"]
    y_pred_shift = model.predict(X_test_shift)
    # probabilidades
    try:
        y_proba_shift = model.predict_proba(X_test_shift)[:, 1]
    except Exception:
        # fallback via decision_function
        def df_to_proba(m, X_):
            s = m.decision_function(X_)
            s = (s - s.min()) / (s.max() - s.min() + 1e-9)
            return s
        y_proba_shift = df_to_proba(model, X_test_shift)

    res["generalization"] = {
        "Accuracy": accuracy_score(y_test, y_pred_shift),
        "F1": f1_score(y_test, y_pred_shift),
        "Recall": recall_score(y_test, y_pred_shift),
        "ROC_AUC": roc_auc_score(y_test, y_proba_shift),
    }

# %% [markdown]
# ## 8. Tabela comparativa final
# - Traz métricas de treino, teste, validação cruzada (média ± desvio), AUC e generalização

# %%
def fmt_cv(cv_dict):
    return {
        "CV_Acc_mean": cv_dict["accuracy"]["mean"],
        "CV_Acc_std": cv_dict["accuracy"]["std"],
        "CV_F1_mean": cv_dict["f1"]["mean"],
        "CV_F1_std": cv_dict["f1"]["std"],
        "CV_Recall_mean": cv_dict["recall"]["mean"],
        "CV_Recall_std": cv_dict["recall"]["std"],
        "CV_AUC_mean": cv_dict["roc_auc"]["mean"],
        "CV_AUC_std": cv_dict["roc_auc"]["std"],
    }

rows = []
for res in results:
    r = {
        "Modelo": res["name"],
        "Train_Acc": res["train"]["Accuracy"],
        "Train_F1": res["train"]["F1"],
        "Train_Recall": res["train"]["Recall"],
        "Train_AUC": res["train"]["ROC_AUC"],
        "Test_Acc": res["test"]["Accuracy"],
        "Test_F1": res["test"]["F1"],
        "Test_Recall": res["test"]["Recall"],
        "Test_AUC": res["test"]["ROC_AUC"],
        "Gen_Acc": res["generalization"]["Accuracy"],
        "Gen_F1": res["generalization"]["F1"],
        "Gen_Recall": res["generalization"]["Recall"],
        "Gen_AUC": res["generalization"]["ROC_AUC"],
    }
    r.update(fmt_cv(res["cv"]))
    rows.append(r)

summary_df = pd.DataFrame(rows)
summary_df = summary_df.sort_values(by="Test_AUC", ascending=False)
pd.options.display.float_format = "{:,.4f}".format
summary_df

# %% [markdown]
# ## 9. Interpretação e conclusões
# - Analisamos:
#   - Diferença treino × teste (overfitting se treino muito maior; underfitting se ambos baixos)
#   - Estabilidade na validação cruzada (médias altas e desvios baixos = robustez)
#   - AUC das curvas ROC
#   - Teste de generalização (manutenção das métricas sob perturbação)

# %%
def print_conclusions(df):
    print("=== Conclusões ===")
    best_auc_model = df.iloc[0]["Modelo"]
    print(f"- Melhor AUC no teste: {best_auc_model}")

    for _, row in df.iterrows():
        name = row["Modelo"]
        train_auc = row["Train_AUC"]
        test_auc = row["Test_AUC"]
        diff_auc = train_auc - test_auc

        train_acc = row["Train_Acc"]
        test_acc = row["Test_Acc"]
        diff_acc = train_acc - test_acc

        cv_mean = row["CV_AUC_mean"]
        cv_std = row["CV_AUC_std"]

        gen_auc = row["Gen_AUC"]

        status = []
        # Overfitting se diferença significativa
        if diff_acc > 0.10 or diff_auc > 0.10:
            status.append("possível overfitting")
        # Underfitting se ambos baixos (critérios simples)
        if (train_acc < 0.65 and test_acc < 0.65) or (train_auc < 0.65 and test_auc < 0.65):
            status.append("possível underfitting")
        if not status:
            status.append("equilíbrio razoável")

        print(f"\nModelo: {name}")
        print(f"  - Train vs Test (Accuracy): {train_acc:.3f} vs {test_acc:.3f} (Δ={diff_acc:.3f})")
        print(f"  - Train vs Test (AUC): {train_auc:.3f} vs {test_auc:.3f} (Δ={diff_auc:.3f})")
        print(f"  - CV AUC: média={cv_mean:.3f}, desvio={cv_std:.3f} (estabilidade {'boa' if cv_std < 0.02 else 'moderada' if cv_std < 0.05 else 'baixa'})")
        print(f"  - Generalização (AUC teste perturbado): {gen_auc:.3f}")
        print(f"  - Interpretação: {', '.join(status)}")

print_conclusions(summary_df)

# 
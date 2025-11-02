import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

RANDOM_STATE = 42
TEST_SIZE = 0.3

# ---------------------------------------------------------------
# 1) Carregar base Adult
# ---------------------------------------------------------------
adult = fetch_openml("adult", version=2, as_frame=True)
X = adult.data
y = (adult.target == ">50K").astype(int)  # 1= >50K, 0= <=50K

print("Instâncias:", X.shape[0], "| Features:", X.shape[1])
print("Distribuição classes:", np.bincount(y))

# ---------------------------------------------------------------
# 2) Separar colunas numéricas e categóricas
# ---------------------------------------------------------------
cat_cols = X.select_dtypes(include="category").columns
num_cols = X.select_dtypes(exclude="category").columns

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# ---------------------------------------------------------------
# 3) Split treino/teste
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------
# 4) Definir modelos
# ---------------------------------------------------------------
modelos = [
    ("Regressão Logística", Pipeline([
        ("prep", preprocess),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])),
    ("SVM Linear", Pipeline([
        ("prep", preprocess),
        ("clf", LinearSVC(max_iter=5000, class_weight="balanced"))
    ])),
    ("Naive Bayes", Pipeline([
        ("prep", preprocess),
        ("clf", GaussianNB())
    ])),
    ("k-NN (k=5)", Pipeline([
        ("prep", preprocess),
        ("clf", KNeighborsClassifier(n_neighbors=5))
    ])),
    ("MLP", Pipeline([
        ("prep", preprocess),
        ("clf", MLPClassifier(hidden_layer_sizes=(32,), max_iter=200,
                              early_stopping=True, random_state=RANDOM_STATE))
    ])),
]

# ---------------------------------------------------------------
# 5) Avaliar modelos
# ---------------------------------------------------------------
resultados = []
for nome, modelo in modelos:
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    resultados.append({"Modelo": nome, "Acc": acc, "Prec": prec, "Recall": rec, "F1": f1})
    print(f"\n=== {nome} ===")
    print(f"Acurácia: {acc:.3f} | Precisão: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")
    print("Matriz de confusão:\n", cm)

df_resultados = pd.DataFrame(resultados)
print("\n=== Comparação geral ===")
print(df_resultados.sort_values("F1", ascending=False))

# ---------------------------------------------------------------
# 6) Fronteiras de decisão em 2D (usando 2 features numéricas)
# ---------------------------------------------------------------
# Escolher duas features numéricas: age e hours-per-week
X2 = X[["age", "hours-per-week"]].values
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

def plot_fronteira(modelo, X_all, X_set, y_set, title):
    h = 1.0
    x_min, x_max = X_all[:,0].min()-1, X_all[:,0].max()+1
    y_min, y_max = X_all[:,1].min()-1, X_all[:,1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = modelo.predict(grid).reshape(xx.shape)
    plt.figure(figsize=(6,4.5))
    plt.contourf(xx, yy, Z, alpha=0.2)
    plt.scatter(X_set[:,0], X_set[:,1], c=y_set, edgecolors="k", s=10)
    plt.xlabel("Age")
    plt.ylabel("Hours per week")
    plt.title(title)
    plt.show()

# Rodar fronteiras para cada modelo
for nome, modelo in modelos:
    # Aqui usamos só as duas features numéricas
    pipe2 = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", modelo.named_steps["clf"])
    ])
    pipe2.fit(X2_train, y2_train)
    plot_fronteira(pipe2, X2, X2_train, y2_train, f"{nome} - Treino")
    plot_fronteira(pipe2, X2, X2_test, y2_test, f"{nome} - Teste")

#- O Adult é desbalanceado (menos pessoas com renda >50K). Por isso, class_weight="balanced" ajuda RL e SVM.
#- O Naive Bayes pode ter desempenho mais fraco porque os atributos não seguem bem a suposição de normalidade.
#- O MLP e o k-NN podem capturar padrões mais complexos, mas podem ser mais lentos.

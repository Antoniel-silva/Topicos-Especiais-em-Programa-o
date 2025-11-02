import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
# 1) Carregar base e definir binária
# ---------------------------------------------------------------
data = fetch_covtype()
X = data.data
y = (data.target == 1).astype(int)  # 1=Spruce/Fir -> positivo

print("Instâncias:", X.shape[0], "| Features:", X.shape[1])
print("Distribuição classes:", np.bincount(y))

# Para acelerar em sala de aula, podemos usar uma amostra menor
# X, _, y, _ = train_test_split(X, y, train_size=50000, stratify=y, random_state=RANDOM_STATE)

# ---------------------------------------------------------------
# 2) Split treino/teste
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------
# 3) Definir modelos
# ---------------------------------------------------------------
modelos = [
    ("Regressão Logística", Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])),
    ("SVM Linear", Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LinearSVC(max_iter=5000, class_weight="balanced"))
    ])),
    ("Naive Bayes", GaussianNB()),
    ("k-NN (k=5)", Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", KNeighborsClassifier(n_neighbors=5))
    ])),
    ("MLP", Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", MLPClassifier(hidden_layer_sizes=(32,), max_iter=200,
                              early_stopping=True, random_state=RANDOM_STATE))
    ])),
]

# ---------------------------------------------------------------
# 4) Avaliar modelos
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
# 5) Fronteiras de decisão em 2D (usando 2 features)
# ---------------------------------------------------------------
# Escolher duas features (ex: Elevation=0, Horizontal_Distance_To_Roadways=10)
X2 = X[:, [0, 10]]
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

def plot_fronteira(modelo, X_all, X_set, y_set, title):
    h = 500  # passo da grade (ajustado para escala do Covertype)
    x_min, x_max = X_all[:,0].min()-100, X_all[:,0].max()+100
    y_min, y_max = X_all[:,1].min()-100, X_all[:,1].max()+100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = modelo.predict(grid).reshape(xx.shape)
    plt.figure(figsize=(6,4.5))
    plt.contourf(xx, yy, Z, alpha=0.2)
    plt.scatter(X_set[:,0], X_set[:,1], c=y_set, edgecolors="k", s=10)
    plt.xlabel("Elevation")
    plt.ylabel("Horizontal_Distance_To_Roadways")
    plt.title(title)
    plt.show()

# Rodar fronteiras para cada modelo
for nome, modelo in modelos:
    modelo.fit(X2_train, y2_train)
    plot_fronteira(modelo, X2, X2_train, y2_train, f"{nome} - Treino")
    plot_fronteira(modelo, X2, X2_test, y2_test, f"{nome} - Teste")


#- Modelos lineares tendem a se sair bem, mas podem errar em regiões complexas.
#- Naive Bayes pode ser mais fraco se as features não seguirem bem a suposição de normalidade.
#- k-NN e MLP podem capturar padrões mais complexos, mas podem sofrer com tempo de treino ou overfitting.

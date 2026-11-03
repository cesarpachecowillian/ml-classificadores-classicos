#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atividade 2: Classificação e Avaliação com Validação Cruzada (5 folds)
Disciplina: Aprendizagem de Máquina (Especialização em IA e Data Science, PUCPR)
Aluno: César Willian Pacheco

"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, ConfusionMatrixDisplay, classification_report)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, BaggingClassifier,
                              GradientBoostingClassifier, VotingClassifier)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ----------------------------------------------------------------------------------------
# 1) Leitura da base (código do enunciado)
# ----------------------------------------------------------------------------------------
data = pd.read_csv("heart_failure_clinical_records_dataset.csv")
val = data.values
att = data.columns
X = val[:, 0:12]
y = val[:, 12].astype(int)

print("=" * 78)
print("BASE DE DADOS: heart_failure_clinical_records_dataset.csv")
print("=" * 78)
print(f"Número de instâncias ............: {X.shape[0]}")
print(f"Número de atributos de entrada ..: {X.shape[1]}")
print(f"Atributos: {list(att[:12])}")
classes, counts = np.unique(y, return_counts=True)
for c, n in zip(classes, counts):
    print(f"Classe {c} ({'infarto' if c == 1 else 'normal '}): {n} exemplos ({100*n/len(y):.1f}%)")

# ----------------------------------------------------------------------------------------
# 2) Validação cruzada estratificada com 5 folds
# ----------------------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ----------------------------------------------------------------------------------------
# 3) Modelos e grades de hiperparâmetros
# ----------------------------------------------------------------------------------------
def scaled(clf):
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

models = {
    # ---- caixa-branca ----
    "Arvore de Decisao": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Naive Bayes": GaussianNB(),
    # ---- caixa-preta ----
    "KNN": scaled(KNeighborsClassifier()),
    "SVM": scaled(SVC(random_state=RANDOM_STATE)),
    "MLP": scaled(MLPClassifier(max_iter=2000, random_state=RANDOM_STATE)),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
    "AdaBoost": AdaBoostClassifier(random_state=RANDOM_STATE),
    "Bagging (Arvores)": BaggingClassifier(estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
                                           random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "Voting (RF+SVM+KNN)": VotingClassifier(estimators=[
        ("rf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
        ("svm", scaled(SVC(probability=True, random_state=RANDOM_STATE))),
        ("knn", scaled(KNeighborsClassifier(n_neighbors=7))),
    ]),
}

param_grids = {
    "Arvore de Decisao": {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 3, 4, 5, 8],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 3, 5, 10],
    },
    "Naive Bayes": {"var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-1]},
    "KNN": {
        "clf__n_neighbors": [1, 3, 5, 7, 9, 11, 15],
        "clf__weights": ["uniform", "distance"],
        "clf__p": [1, 2],
    },
    "SVM": {
        "clf__C": [0.1, 1, 10, 100],
        "clf__kernel": ["rbf", "linear"],
        "clf__gamma": ["scale", 0.01, 0.001],
        "clf__class_weight": [None, "balanced"],
    },
    "MLP": {
        "clf__hidden_layer_sizes": [(8,), (16,), (32,), (16, 8)],
        "clf__activation": ["relu", "tanh"],
        "clf__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
    },
    "Random Forest": {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 4, 8],
        "class_weight": [None, "balanced"],
    },
    "AdaBoost": {"n_estimators": [50, 100, 200], "learning_rate": [0.1, 0.5, 1.0]},
    "Bagging (Arvores)": {"n_estimators": [50, 100, 200], "max_samples": [0.5, 0.75, 1.0]},
    "Gradient Boosting": {"n_estimators": [50, 100, 200], "learning_rate": [0.05, 0.1],
                          "max_depth": [2, 3]},
    "Voting (RF+SVM+KNN)": {"voting": ["hard", "soft"]},
}

caixa_branca = {"Arvore de Decisao", "Naive Bayes"}

# ----------------------------------------------------------------------------------------
# 4) Busca de hiperparâmetros + avaliação por validação cruzada (5 folds)
# ----------------------------------------------------------------------------------------
results, best_models, matrices = [], {}, {}

for name, model in models.items():
    print("\n" + "-" * 78)
    print(f"Modelo: {name}")
    gs = GridSearchCV(model, param_grids[name], scoring="f1", cv=cv, n_jobs=-1, refit=True)
    gs.fit(X, y)
    best = gs.best_estimator_
    best_models[name] = best
    print("Melhores hiperparâmetros:", gs.best_params_)

    # predições fora da amostra, fold a fold, com os mesmos 5 folds
    y_pred = cross_val_predict(best, X, y, cv=cv)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)                  # F1 da classe positiva (infarto)
    f1w = f1_score(y, y_pred, average="weighted")
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    cm = confusion_matrix(y, y_pred, labels=[1, 0])   # linhas/colunas: [infarto, normal]
    matrices[name] = cm
    print(f"Taxa de acerto = {100*acc:.2f}%  |  F1(infarto) = {f1:.4f}  |  F1 ponderado = {f1w:.4f}"
          f"  |  Precisao = {prec:.4f}  |  Revocacao = {rec:.4f}")
    print("Matriz de confusao [linhas = real, colunas = predito] ordem [infarto, normal]:")
    print(cm)
    results.append({"modelo": name,
                    "tipo": "caixa-branca" if name in caixa_branca else "caixa-preta",
                    "taxa_acerto_%": round(100 * acc, 2),
                    "f1_infarto": round(f1, 4), "f1_ponderado": round(f1w, 4),
                    "precisao": round(prec, 4), "revocacao": round(rec, 4),
                    "VP": int(cm[0, 0]), "FN": int(cm[0, 1]), "FP": int(cm[1, 0]), "VN": int(cm[1, 1]),
                    "hiperparametros": str(gs.best_params_)})

df = pd.DataFrame(results).sort_values("f1_infarto", ascending=False)
df.to_csv("resultados_atividade2.csv", index=False)

print("\n" + "=" * 78)
print("RESUMO (ordenado por F1 da classe infarto)")
print("=" * 78)
print(df[["modelo", "tipo", "taxa_acerto_%", "f1_infarto", "precisao", "revocacao"]].to_string(index=False))

# ----------------------------------------------------------------------------------------
# 5) Melhor caixa-preta e melhor caixa-branca
# ----------------------------------------------------------------------------------------
for tipo in ["caixa-preta", "caixa-branca"]:
    linha = df[df["tipo"] == tipo].iloc[0]
    name = linha["modelo"]
    cm = matrices[name]
    VP, FN, FP, VN = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    print("\n" + "=" * 78)
    print(f"MELHOR MODELO {tipo.upper()}: {name}")
    print(f"Hiperparâmetros: {linha['hiperparametros']}")
    print(f"Taxa de acerto = {linha['taxa_acerto_%']:.2f}%   F1 = {linha['f1_infarto']:.4f}")
    print(f"VP = {VP}  FP = {FP}  VN = {VN}  FN = {FN}")
    print(f"Taxa de VP (sensibilidade/recall)  = VP/(VP+FN) = {VP/(VP+FN):.4f}")
    print(f"Taxa de FP                          = FP/(FP+VN) = {FP/(FP+VN):.4f}")
    print(f"Taxa de VN (especificidade)         = VN/(VN+FP) = {VN/(VN+FP):.4f}")
    print(f"Taxa de FN                          = FN/(FN+VP) = {FN/(FN+VP):.4f}")
    print(f"Taxa de acerto por classe: infarto = {100*VP/(VP+FN):.2f}%   normal = {100*VN/(VN+FP):.2f}%")
    print(classification_report(y, cross_val_predict(best_models[name], X, y, cv=cv),
                                target_names=["normal (0)", "infarto (1)"]))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["infarto (1)", "normal (0)"])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    disp.plot(ax=ax, values_format="d", cmap="Greys", colorbar=False)
    ax.set_title(f"{name}: validação cruzada com 5 folds")
    ax.set_xlabel("Classe predita"); ax.set_ylabel("Classe real")
    plt.tight_layout()
    fname = "matriz_confusao_" + name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "_") + ".png"
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"Matriz de confusão salva em {fname}")

# ----------------------------------------------------------------------------------------
# 6) Importância dos atributos segundo a Árvore de Decisão (questão B.8)
# ----------------------------------------------------------------------------------------
tree = best_models["Arvore de Decisao"]
imp = pd.Series(tree.feature_importances_, index=att[:12]).sort_values(ascending=False)
print("\n" + "=" * 78)
print("IMPORTÂNCIA DOS ATRIBUTOS (Árvore de Decisão treinada na base completa)")
print("=" * 78)
print(imp.round(4).to_string())
print(f"\nAtributo mais importante: {imp.index[0]}  (importância = {imp.iloc[0]:.4f})")
print(f"Atributo da raiz da árvore: {att[tree.tree_.feature[0]]}  (limiar = {tree.tree_.threshold[0]:.2f})")

fig, ax = plt.subplots(figsize=(6.5, 3.8))
imp[::-1].plot.barh(ax=ax, color="0.35")
ax.set_xlabel("Importância (redução de impureza)")
ax.set_title("Importância dos atributos na Árvore de Decisão")
plt.tight_layout()
plt.savefig("importancia_atributos_arvore.png", dpi=200)
plt.close()
print("Gráfico salvo em importancia_atributos_arvore.png")

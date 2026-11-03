#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atividade 3: Classificação e Avaliação com Holdout 60/40
Disciplina: Aprendizagem de Máquina (Especialização em IA e Data Science, PUCPR)
Aluno: César Willian Pacheco

"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (VotingClassifier, BaggingClassifier, AdaBoostClassifier,
                              RandomForestClassifier)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ----------------------------------------------------------------------------------------
# 1) Leitura da base
# ----------------------------------------------------------------------------------------
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
try:
    import urllib.request
    raw_data = urllib.request.urlopen(url, timeout=20)
    dataset = np.loadtxt(raw_data, delimiter=",")
    print("Base carregada a partir da URL do UCI.")
except Exception as e:
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "glass.data")
    dataset = np.loadtxt(local, delimiter=",")
    print(f"Sem acesso à URL ({type(e).__name__}); base carregada do arquivo local glass.data.")

X = dataset[:, 1:10]          # ignora o ID
y = dataset[:, 10].astype(int)
LABELS = [1, 2, 3, 5, 6, 7]   # "123567": a classe 4 não possui exemplos

print("=" * 78)
print("BASE GLASS")
print("=" * 78)
print(f"Instâncias: {X.shape[0]}   Atributos de entrada: {X.shape[1]}")
cls, cnt = np.unique(y, return_counts=True)
print("Distribuição das classes:", dict(zip(cls.tolist(), cnt.tolist())))

# ----------------------------------------------------------------------------------------
# 2) Holdout 60/40 estratificado
# ----------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.40, random_state=RANDOM_STATE, stratify=y)
print(f"Treinamento: {X_train.shape[0]} instâncias | Teste: {X_test.shape[0]} instâncias")
cls, cnt = np.unique(y_test, return_counts=True)
print("Distribuição das classes no teste:", dict(zip(cls.tolist(), cnt.tolist())))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

def scaled(clf):
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

# ----------------------------------------------------------------------------------------
# 3) Classificadores monolíticos e grades de hiperparâmetros
# ----------------------------------------------------------------------------------------
monoliticos = {
    "Arvore de Decisao": (DecisionTreeClassifier(random_state=RANDOM_STATE), {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 4, 6, 8, 12],
        "min_samples_split": [2, 4, 8],
        "min_samples_leaf": [1, 2, 4]}),
    "KNN": (scaled(KNeighborsClassifier(weights="uniform")), {
        "clf__n_neighbors": [1, 3, 5, 7, 9],
        "clf__p": [1, 2]}),
    "KNN weighted": (scaled(KNeighborsClassifier(weights="distance")), {
        "clf__n_neighbors": [1, 3, 5, 7, 9],
        "clf__p": [1, 2]}),
    "Naive Bayes": (GaussianNB(), {
        "var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-1]}),
    "MLP": (scaled(MLPClassifier(max_iter=3000, random_state=RANDOM_STATE)), {
        "clf__hidden_layer_sizes": [(16,), (32,), (64,), (32, 16)],
        "clf__activation": ["relu", "tanh"],
        "clf__alpha": [1e-4, 1e-3, 1e-2]}),
    "SVM": (scaled(SVC(random_state=RANDOM_STATE)), {
        "clf__C": [1, 10, 100, 1000],
        "clf__kernel": ["rbf", "linear", "poly"],
        "clf__gamma": ["scale", 0.1, 0.01]}),
}

results, best, preds = [], {}, {}

def avalia(name, estimator, grid, grupo):
    gs = GridSearchCV(estimator, grid, scoring="accuracy", cv=cv, n_jobs=-1)
    gs.fit(X_train, y_train)
    model = gs.best_estimator_
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    best[name], preds[name] = model, y_pred
    print(f"\n[{grupo}] {name}")
    print(f"  Melhores hiperparâmetros (CV 5 folds no treino): {gs.best_params_}")
    print(f"  Acurácia média na validação cruzada (treino): {100*gs.best_score_:.2f}%")
    print(f"  TAXA DE ACERTO NO TESTE (holdout 40%): {100*acc:.2f}%")
    results.append({"grupo": grupo, "classificador": name, "taxa_acerto_%": round(100*acc, 2),
                    "hiperparametros": str(gs.best_params_)})
    return model

for name, (est, grid) in monoliticos.items():
    avalia(name, est, grid, "Monolitico")

# ----------------------------------------------------------------------------------------
# 4) Ensembles
# ----------------------------------------------------------------------------------------
# 4a) Heterogêneos: reutilizam os melhores monolíticos já ajustados (KNN weighted, SVM, MLP,
#     Árvore) -- para o voto "soft" o SVM precisa de probability=True.
membros_hard = [("dt", best["Arvore de Decisao"]), ("knnw", best["KNN weighted"]),
                ("svm", best["SVM"]), ("mlp", best["MLP"])]
svm_proba = scaled(SVC(probability=True, random_state=RANDOM_STATE,
                       **{k.replace("clf__", ""): v for k, v in
                          best["SVM"].get_params().items()
                          if k in ("clf__C", "clf__kernel", "clf__gamma")}))
membros_soft = [("dt", best["Arvore de Decisao"]), ("knnw", best["KNN weighted"]),
                ("svm", svm_proba), ("mlp", best["MLP"])]

vc_hard = VotingClassifier(estimators=membros_hard, voting="hard")
vc_soft = VotingClassifier(estimators=membros_soft, voting="soft")
for name, model in [("Ensemble heterogeneo com voto majoritario (DT+KNNw+SVM+MLP)", vc_hard),
                    ("Ensemble heterogeneo com soma de probabilidades soft (DT+KNNw+SVM+MLP)", vc_soft)]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    best[name], preds[name] = model, y_pred
    print(f"\n[Ensemble] {name}\n  TAXA DE ACERTO NO TESTE (holdout 40%): {100*acc:.2f}%")
    results.append({"grupo": "Ensemble", "classificador": name, "taxa_acerto_%": round(100*acc, 2),
                    "hiperparametros": "membros = melhores monoliticos; voting=" + model.voting})

# 4b) Homogêneos
homogeneos = {
    "Bagging (Arvores)": (BaggingClassifier(estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
                                            random_state=RANDOM_STATE), {
        "n_estimators": [50, 100, 200, 500],
        "max_samples": [0.5, 0.75, 1.0]}),
    "AdaBoost (Arvores)": (AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE),
                                              random_state=RANDOM_STATE), {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.1, 0.5, 1.0]}),
    "Random Forest": (RandomForestClassifier(random_state=RANDOM_STATE), {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 8, 16],
        "max_features": ["sqrt", None]}),
}
for name, (est, grid) in homogeneos.items():
    avalia(name, est, grid, "Ensemble")

# ----------------------------------------------------------------------------------------
# 5) Tabela final, melhor resultado, matriz de confusão e taxa de acerto por classe
# ----------------------------------------------------------------------------------------
df = pd.DataFrame(results)
df.to_csv("resultados_atividade3.csv", index=False)
print("\n" + "=" * 78)
print("TABELA DE RESULTADOS (Holdout 60/40)")
print("=" * 78)
print(df[["grupo", "classificador", "taxa_acerto_%"]].to_string(index=False))

best_row = df.loc[df["taxa_acerto_%"].idxmax()]
bname = best_row["classificador"]
print("\n" + "=" * 78)
print(f"MELHOR RESULTADO: {bname} -> {best_row['taxa_acerto_%']:.2f}%")
print(f"Parâmetros: {best_row['hiperparametros']}")
print("=" * 78)

y_pred = preds[bname]
cm = confusion_matrix(y_test, y_pred, labels=LABELS)
print("Matriz de confusão (linhas = classe real, colunas = predita), classes 1 2 3 5 6 7:")
print(cm)
print(classification_report(y_test, y_pred, labels=LABELS, zero_division=0))

print("Taxa de acerto por classe:")
acc_cls = {}
for i, c in enumerate(LABELS):
    tot = cm[i].sum(); ok = cm[i, i]
    acc_cls[c] = ok / tot if tot else float("nan")
    print(f"  Classe {c}: {ok}/{tot} = {100*acc_cls[c]:.2f}%  (erros: {tot-ok})")
pior = min(acc_cls, key=lambda c: acc_cls[c])
mais_erros = max(LABELS, key=lambda c: cm[LABELS.index(c)].sum() - cm[LABELS.index(c), LABELS.index(c)])
print(f"\nClasse com MAIS ERROS (em quantidade): {mais_erros} -> taxa de acerto {100*acc_cls[mais_erros]:.2f}%")
print(f"Classe com MENOR taxa de acerto: {pior} -> {100*acc_cls[pior]:.2f}%")

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
fig, ax = plt.subplots(figsize=(5.2, 4.6))
disp.plot(ax=ax, values_format="d", cmap="Greys", colorbar=False)
ax.set_title(f"{bname}\nHoldout 60/40, teste com {len(y_test)} instâncias", fontsize=9)
ax.set_xlabel("Classe predita"); ax.set_ylabel("Classe real")
plt.tight_layout()
plt.savefig("matriz_confusao_melhor.png", dpi=200)
plt.close()
print("Matriz de confusão salva em matriz_confusao_melhor.png")

# Proyecto electrico: Implementación de algoritmos Random Forest y Naive Bayes
# para clasificación multietiqueta en la detección de fallas:
# Desarrollo de una métrica de evaluación comparativa.
# Estudiante: Ebandry Calderón Araya.
# Código que permite entrenar y evaluar los modelos RF y NB con un
# dataset sintético multietiqueta.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, hamming_loss,
    roc_curve, auc, f1_score, jaccard_score,
    average_precision_score,
    label_ranking_average_precision_score, label_ranking_loss,
    brier_score_loss
)

# se carga dataset sintetico 
df = pd.read_csv("datos_mult_sint_final.csv")

feature_cols = [
    "BUS10Va","BUS10Vb","BUS10Vc",
    "L4Ia_FLT","L4Ib_FLT","L4Ic_FLT",
    "F_index","ratioA","ratioB","ratioC"
]
label_cols = ["FaseA","FaseB","FaseC"]

X = df[feature_cols].copy()
Y = df[label_cols].astype(int).copy()

print(f"Total muestras: {len(df)}")
for c in label_cols:
    v = Y[c].value_counts()
    print(f"{c}: 0={v.get(0,0)}, 1={v.get(1,0)}")

# distribucion de combinaciones de etiquetas
combos_all = Y.apply(lambda r: "".join(map(str, r.values)), axis=1)
print("\nDistribucion de combinaciones:")
print(combos_all.value_counts())

# train / test por combinacion de etiquetas
combos = Y.apply(tuple, axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42, stratify=combos
)
print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

# escalado
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# para metricas multilabel
Y_true_bin = Y_test[label_cols].values

# RANDOM FOREST
rf_base = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=42
)
rf = MultiOutputClassifier(rf_base)

print("\nRandom Forest")
rf.fit(X_train_s, Y_train)
Y_pred_rf = rf.predict(X_test_s)

# metricas globales RF
print("\nMetricas basicas RF")

# exact-match y hamming
exact_rf = accuracy_score(Y_test, Y_pred_rf)
hamm_rf  = hamming_loss(Y_test, Y_pred_rf)

# jaccard solo en muestras con una etiqueta verdadera
Y_true_bin = Y_test[label_cols].values
mask_fault_rf = (Y_true_bin.sum(axis=1) + Y_pred_rf.sum(axis=1)) > 0

jacc_rf = jaccard_score(
    Y_true_bin[mask_fault_rf],
    Y_pred_rf[mask_fault_rf],
    average="samples",
    zero_division=0
)

print(f"Exact-match accuracy : {exact_rf:.4f}")
print(f"Hamming loss         : {hamm_rf:.4f}")
print(f"Jaccard (samples)    : {jacc_rf:.4f}")

print(classification_report(Y_test, Y_pred_rf,
                            target_names=label_cols,
                            zero_division=0))

# curvas ROC y AUPRC por fase
fig, axes = plt.subplots(1, 3, figsize=(15,4))
fig.suptitle("Curvas ROC por fase RF")

rf_probas = [est.predict_proba(X_test_s)[:,1] for est in rf.estimators_]
rf_scores = np.column_stack(rf_probas)

ap_rf = []

for i, col in enumerate(label_cols):
    y_true = Y_test[col].values
    y_score = rf_probas[i]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    ap = average_precision_score(y_true, y_score)
    ap_rf.append(ap)

    axes[i].plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    axes[i].plot([0,1],[0,1],'--',color="gray")
    axes[i].set_title(col)
    axes[i].set_xlabel("FPR")
    axes[i].set_ylabel("TPR")
    axes[i].legend(loc="lower right")

plt.tight_layout()
plt.show()

print("\nAUPRC por fase RF:")
for col, ap in zip(label_cols, ap_rf):
    print(f"  {col}: {ap:.3f}")

# metricas de ranking probabilisticas
lrap_rf  = label_ranking_average_precision_score(Y_true_bin, rf_scores)
rloss_rf = label_ranking_loss(Y_true_bin, rf_scores)

print("\nRanking metricas RF:")
print(f"  LRAP         : {lrap_rf:.4f}")
print(f"  Ranking loss : {rloss_rf:.4f}")

# brier score por fase 
print("\nBrier score por fase RF:")
for j, col in enumerate(label_cols):
    bs = brier_score_loss(Y_test[col].values, rf_scores[:, j])
    print(f"  {col}: {bs:.4f}")

# robustez al ruido 
noise_levels = [0.0, 0.01, 0.02, 0.05, 0.10]
accs_rf, f1_rf = [], []

for sigma in noise_levels:
    X_noisy = X_test_s + np.random.normal(0, sigma, X_test_s.shape)
    Yp = rf.predict(X_noisy)
    accs_rf.append(accuracy_score(Y_test, Yp))
    # F1 macro sobre las 3 etiquetas
    f1_rf.append(np.mean([
        f1_score(Y_test.iloc[:, j], Yp[:, j], zero_division=0)
        for j in range(Y_test.shape[1])
    ]))

plt.figure(figsize=(6,4))
plt.plot(noise_levels, accs_rf, marker="o", label="Acc RF")
plt.plot(noise_levels, f1_rf, marker="s", label="F1 RF")
plt.xlabel("σ ruido")
plt.ylabel("Metrica")
plt.title("Robustez al ruido RF")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

print("\nResumen RF:")
print(f"  Exact-match={exact_rf:.4f} | Jaccard={jacc_rf:.4f} | "
      f"LRAP={lrap_rf:.4f} | RankingLoss={rloss_rf:.4f}")


# NAIVE BAYES
nb_base = GaussianNB()
nb = MultiOutputClassifier(nb_base)

print("\nNaive Bayes")
nb.fit(X_train_s, Y_train)
Y_pred_nb = nb.predict(X_test_s)

# metricas globales NB 
print("\nMetricas basicas NB")

exact_nb = accuracy_score(Y_test, Y_pred_nb)
hamm_nb  = hamming_loss(Y_test, Y_pred_nb)

# Jaccard solo muestras con una etiqueta verdadera
Y_true_bin = Y_test[label_cols].values
mask_fault_nb = (Y_true_bin.sum(axis=1) + Y_pred_nb.sum(axis=1)) > 0

jacc_nb = jaccard_score(
    Y_true_bin[mask_fault_nb],
    Y_pred_nb[mask_fault_nb],
    average="samples",
    zero_division=0
)

print(f"Exact-match accuracy : {exact_nb:.4f}")
print(f"Hamming loss         : {hamm_nb:.4f}")
print(f"Jaccard (samples)    : {jacc_nb:.4f}")
print(classification_report(Y_test, Y_pred_nb,
                            target_names=label_cols,
                            zero_division=0))

# curvas ROC y AUPRC por fase
fig, axes = plt.subplots(1, 3, figsize=(15,4))
fig.suptitle("Curvas ROC por fase NB")

nb_probas = [est.predict_proba(X_test_s)[:,1] for est in nb.estimators_]
nb_scores = np.column_stack(nb_probas)
ap_nb = []

for i, col in enumerate(label_cols):
    y_true = Y_test[col].values
    y_score = nb_probas[i]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    ap = average_precision_score(y_true, y_score)
    ap_nb.append(ap)

    axes[i].plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    axes[i].plot([0,1],[0,1],'--',color="gray")
    axes[i].set_title(col)
    axes[i].set_xlabel("FPR")
    axes[i].set_ylabel("TPR")
    axes[i].legend(loc="lower right")

plt.tight_layout()
plt.show()

print("\nAUPRC por fase NB:")
for col, ap in zip(label_cols, ap_nb):
    print(f"  {col}: {ap:.3f}")

# metricas de ranking probabilisticas
lrap_nb  = label_ranking_average_precision_score(Y_true_bin, nb_scores)
rloss_nb = label_ranking_loss(Y_true_bin, nb_scores)

print("\nRanking metricas NB:")
print(f"  LRAP         : {lrap_nb:.4f}")
print(f"  Ranking loss : {rloss_nb:.4f}")

# brier score por fase 
print("\nBrier score por fase NB:")
for j, col in enumerate(label_cols):
    bs = brier_score_loss(Y_test[col].values, nb_scores[:, j])
    print(f"  {col}: {bs:.4f}")

# robustez al ruido 
accs_nb, f1_nb = [], []

for sigma in noise_levels:
    X_noisy = X_test_s + np.random.normal(0, sigma, X_test_s.shape)
    Yp = nb.predict(X_noisy)
    accs_nb.append(accuracy_score(Y_test, Yp))
    f1_nb.append(np.mean([
        f1_score(Y_test.iloc[:, j], Yp[:, j], zero_division=0)
        for j in range(Y_test.shape[1])
    ]))

plt.figure(figsize=(6,4))
plt.plot(noise_levels, accs_nb, marker="o", label="Acc NB")
plt.plot(noise_levels, f1_nb, marker="s", label="F1 NB")
plt.xlabel("σ ruido")
plt.ylabel("Metrica")
plt.title("Robustez al ruido Naive Bayes")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

print("\nResumen NB:")
print(f"  Exact-match={exact_nb:.4f} | Jaccard={jacc_nb:.4f} | "
      f"LRAP={lrap_nb:.4f} | RankingLoss={rloss_nb:.4f}")

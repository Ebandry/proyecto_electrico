# Proyecto electrico: Implementación de algoritmos Random Forest y Naive Bayes
# para clasificación multietiqueta en la detección de fallas:
# Desarrollo de una métrica de evaluación comparativa.
# Estudiante: Ebandry Calderón Araya.
# Código etiquetador inicial basado en impedancia aparente (Z = V/I). Recibe
# Time, BUS10Va, BUS10Vb, BUS10Vc, L4Ia_FLT, L4Ib_FLT, L4Ic_FLT y genera 
# columnas extra de F_index, ratioA/B/C, Falla_Global, fase_dominante y 
# FaseA/B/C.

import pandas as pd
import numpy as np

INPUT_FILE  = "datos_raw_v1.csv"
OUTPUT_FILE = "datos_etiquetados.csv"

# parametros ajustables
# segundos de operacion normal al inicio
PREFAULT_DURATION_S = 0.15   
# sigmas por debajo del promedio
K_FACTOR             = 3.0   
# duracion minima de falla en ms
MIN_DUR_FAULT_MS     = 5.0   
# exigir un aumento de corriente global
USE_I_THRESHOLD      = True
# numero sigmas por encima del promedio para I_total
I_K_FACTOR           = 3.0   
# evitar divisiones por cero
eps = 1e-9  

# se carga datos crudos
df = pd.read_csv(INPUT_FILE)
print(f"Dataset cargado: {len(df)} muestras")
print("Columnas:", df.columns.tolist())

# columnas entrada
required_cols = ["Time", "BUS10Va", "BUS10Vb", "BUS10Vc",
                 "L4Ia_FLT", "L4Ib_FLT", "L4Ic_FLT"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Falta la columna requerida: {c}")

# ordenar por tiempo
df = df.sort_values("Time").reset_index(drop=True)

# muestreo aproximado
dt = df["Time"].iloc[1] - df["Time"].iloc[0]
fs = 1.0 / dt
print(f"dt ≈ {dt:.6e} s,  fs ≈ {fs:.1f} Hz")

#  indice global F(t) = |V| / |I|
Va = df["BUS10Va"].values
Vb = df["BUS10Vb"].values
Vc = df["BUS10Vc"].values

Ia = df["L4Ia_FLT"].values
Ib = df["L4Ib_FLT"].values
Ic = df["L4Ic_FLT"].values

V_abs_sum = np.abs(Va) + np.abs(Vb) + np.abs(Vc)
I_abs_sum = np.abs(Ia) + np.abs(Ib) + np.abs(Ic)

df["F_index"] = V_abs_sum / (I_abs_sum + eps)

# impedancia aparente de cada fase
df["ratioA"] = np.abs(Va) / (np.abs(Ia) + eps)
df["ratioB"] = np.abs(Vb) / (np.abs(Ib) + eps)
df["ratioC"] = np.abs(Vc) / (np.abs(Ic) + eps)

# corriente total RMS instantanea
df["I_total"] = np.sqrt(Ia**2 + Ib**2 + Ic**2)

# se define ventana de referencia pre-falla
t0 = df["Time"].iloc[0]
t_ref_end = t0 + PREFAULT_DURATION_S

mask_ref = df["Time"] <= t_ref_end
if mask_ref.sum() < 100:
    raise ValueError("La ventana pre-falla es demasiado pequeña, ajustar PREFAULT_DURATION_S.")

F_ref = df.loc[mask_ref, "F_index"]
I_ref = df.loc[mask_ref, "I_total"]

mean_F = F_ref.mean()
std_F  = F_ref.std(ddof=0)

print("\nEstadísticos en ventana pre-falla")
print(f"Ventana pre-falla: desde t={t0:.6f} s hasta t={t_ref_end:.6f} s "
      f"({mask_ref.sum()} muestras)")
print(f"F_index (ref): mean={mean_F:.6e}, std={std_F:.6e}")

threshold_F = mean_F - K_FACTOR * std_F
print(f"Umbral F_index de falla: F_index < {threshold_F:.6e} "
      f"(k={K_FACTOR})")

if USE_I_THRESHOLD:
    mean_I = I_ref.mean()
    std_I  = I_ref.std(ddof=0)
    threshold_I = mean_I + I_K_FACTOR * std_I
    print(f"I_total (ref): mean={mean_I:.6e}, std={std_I:.6e}")
    print(f"Umbral de corriente: I_total > {threshold_I:.6e} "
          f"(k={I_K_FACTOR})")
else:
    threshold_I = None

# deteccion de falla global
cond_F = df["F_index"].values < threshold_F
if USE_I_THRESHOLD:
    cond_I = df["I_total"].values > threshold_I
    mask_fault_raw = cond_F & cond_I
else:
    mask_fault_raw = cond_F

df["Falla_Global_raw"] = mask_fault_raw.astype(int)

# se  filtra persistencia temporal
min_samples = int((MIN_DUR_FAULT_MS / 1000.0) * fs)
min_samples = max(min_samples, 1)
print(f"\nPersistencia mínima de falla: {MIN_DUR_FAULT_MS:.2f} ms "
      f" {min_samples} muestras consecutivas")

mask = df["Falla_Global_raw"].values
filtered_mask = np.zeros_like(mask, dtype=int)

count = 0
for i, val in enumerate(mask):
    if val == 1:
        count += 1
    else:
        if count >= min_samples:
            filtered_mask[i - count:i] = 1
        count = 0

# borde final
if count >= min_samples:
    filtered_mask[len(mask) - count:len(mask)] = 1

df["Falla_Global"] = filtered_mask

# fase dominante por impedancia aparente
# fase_dominante = solo donde hay falla_global
ratios = df[["ratioA", "ratioB", "ratioC"]].copy()
# ratioA/ratioB/ratioC
fase_dom_idx = ratios.idxmin(axis=1)   

# donde no hay falla global se pone NaN
fase_dom_idx = np.where(df["Falla_Global"] == 1, fase_dom_idx, np.nan)
df["fase_dominante"] = fase_dom_idx

# etiquetas binarias por fase
df["FaseA"] = ((df["fase_dominante"] == "ratioA").astype(int))
df["FaseB"] = ((df["fase_dominante"] == "ratioB").astype(int))
df["FaseC"] = ((df["fase_dominante"] == "ratioC").astype(int))

# donde no hay Falla_Global se fuerza 0
df.loc[df["Falla_Global"] == 0, ["FaseA", "FaseB", "FaseC"]] = 0

# resumen estadistico
n_total = len(df)
n_fault = int(df["Falla_Global"].sum())
pct_fault = 100.0 * n_fault / n_total

print("Resumen estadistico etiquetado")
print(f"Total de muestras analizadas: {n_total}")
print(f"Muestras con falla global: {n_fault} ({pct_fault:.2f}%)")
print(f"Muestras sin falla: {n_total - n_fault} ({100.0 - pct_fault:.2f}%)")

for fase in ["FaseA", "FaseB", "FaseC"]:
    n_f = int(df[fase].sum())
    pct = 100.0 * n_f / n_total
    print(f"  - {fase}: {n_f} muestras etiquetadas ({pct:.2f}%)")

# distribucion conjunta FaseA, FaseB, FaseC
print("\nDistribución conjunta de etiquetas (FaseA, FaseB, FaseC):")
print(df[["FaseA", "FaseB", "FaseC"]].value_counts().sort_index())

# se guarda resultado
df.to_csv(OUTPUT_FILE, index=False)

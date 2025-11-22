# Proyecto electrico: Implementación de algoritmos Random Forest y Naive Bayes
# para clasificación multietiqueta en la detección de fallas:
# Desarrollo de una métrica de evaluación comparativa.
# Estudiante: Ebandry Calderón Araya.
# Código que permite generar un dataset sintético multietiqueta, este toma de
# base el dataset raw y la plantilla de falla detectada en la fase C, con 
# esto se inyectan fallas sintéticas para poder entrenar los modelos.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# se carga datos raw y plantilla de falla real
# Time, BUS10V, L4I_FLT
df_raw = pd.read_csv("datos_raw_v1.csv")
# plantilla de Ic en falla real
Ic_tpl = np.load("plantilla_Ic_falla_C.npy")  
tpl_len = len(Ic_tpl)

n = len(df_raw)
print(f"Datos crudos cargados: {n} muestras, longitud plantilla: {tpl_len}")

# indices de falla real en C para no pasar por encima
# se obtuvieron del codigo de plantilla
s_real, e_real = 16192, 16447  
forbidden = (max(0, s_real - 500), min(n - 1, e_real + 500))

df_aug = df_raw.copy()

# parametros globales
# referencia sana para F_index / I_total
N_REF   = 3000   
# margen extra antes de permitir eventos
MARGEN  = 2000   
# margen al final del archivo
BUFFER  = 1000   
# separacion minima entre eventos sintéticos
SEP_EVT = 150    

# funcion escoger inicio valido de un evento sintético
# se respeta ventana de referencia y margen, no pisar 
# la falla real y no se solapa con eventos ya creados
def elegir_inicio_valido(n_samples,
                         tpl_len,
                         forbidden_range,
                         eventos,
                         n_ref=N_REF,
                         margen=MARGEN,
                         buffer=BUFFER,
                         sep_evt=SEP_EVT,
                         max_tries=10000):
    lo_forb, hi_forb = forbidden_range

    for _ in range(max_tries):
        start = np.random.randint(n_ref + margen,
                                  n_samples - tpl_len - buffer)
        end = start + tpl_len - 1

        # no pisar ni acercarse demasiado a la falla real
        if (lo_forb - tpl_len) <= start <= (hi_forb + tpl_len):
            continue

        # no solapar con otros eventos sinteticos
        solapa = False
        for ev in eventos:
            s2 = ev["start"] - sep_evt
            e2 = ev["end"]   + sep_evt
            if not (end < s2 or start > e2):
                solapa = True
                break
        if solapa:
            continue

        return start

    raise RuntimeError("No se pudo encontrar un inicio valido para el evento sintetico")

# eventos sinteticos mono, bi y trifasicos
eventos_plan = [
    ["A"], ["A"], ["A"], ["A"],      # 4 fallas A
    ["B"], ["B"], ["B"], ["B"],      # 4 B
    ["C"], ["C"], ["C"], ["C"],      # 4 C
    ["A", "B"], ["A", "C"],          # 2 bifasicas
    ["B", "C"],                      # 1 bifasica
    ["A", "B", "C"]                  # 1 trifasica
]
num_fallas = len(eventos_plan)

# inyeccion fallas sinteticas
np.random.seed(42)
eventos = []

for k, fases_ev in enumerate(eventos_plan, start=1):
    start = elegir_inicio_valido(n, tpl_len, forbidden, eventos)
    end   = start + tpl_len

    # escalado alrededor de la plantilla real
    ki = np.random.uniform(0.8, 1.2)
    sigma_noise = 0.03 * np.std(Ic_tpl)

    for phase in fases_ev:
        col_I = f"L4I{phase.lower()}_FLT"
        noise = np.random.normal(0, sigma_noise, size=tpl_len)
        df_aug.loc[start:end-1, col_I] = ki * Ic_tpl + noise

    eventos.append({
        "phases": fases_ev,
        "start": start,
        "end":   end
    })

    print(f"Falla sintetica #{k} en fases {fases_ev} "
          f"entre muestras {start}-{end-1} (ki≈{ki:.2f})")

# etiquetador fisico
def etiquetar_fisico(df,
                     N_ref=3000,
                     kF=3.0,
                     kI=3.0,
                     min_dur_ms=5.0):
    eps = 1e-9

    dt = df["Time"].iloc[1] - df["Time"].iloc[0]
    fs = 1.0 / dt

    Va = df["BUS10Va"].values
    Vb = df["BUS10Vb"].values
    Vc = df["BUS10Vc"].values
    Ia = df["L4Ia_FLT"].values
    Ib = df["L4Ib_FLT"].values
    Ic = df["L4Ic_FLT"].values

    # indice de impedancia aparente
    F_index = (np.abs(Va) + np.abs(Vb) + np.abs(Vc)) / \
              (np.abs(Ia) + np.abs(Ib) + np.abs(Ic) + eps)
    df["F_index"] = F_index

    # corriente total
    I_total = np.abs(Ia) + np.abs(Ib) + np.abs(Ic)
    df["I_total"] = I_total

    # umbrales en ventana de referencia
    F_ref = F_index[:N_ref]
    I_ref = I_total[:N_ref]

    thr_F = F_ref.mean() - kF * F_ref.std()
    thr_I = I_ref.mean() + kI * I_ref.std()

    mask = (F_index < thr_F) & (I_total > thr_I)

    # persistencia minima
    min_len = int(min_dur_ms / 1000.0 * fs)

    filt = np.zeros_like(mask, dtype=int)
    count = 0
    for i, m in enumerate(mask):
        if m:
            count += 1
        elif count > 0:
            if count >= min_len:
                filt[i-count:i] = 1
            count = 0
    if count >= min_len:
        filt[len(mask)-count:] = 1

    df["Falla_Global"] = filt

    # ratios por fase 
    ratioA = np.abs(Va) / (np.abs(Ia) + eps)
    ratioB = np.abs(Vb) / (np.abs(Ib) + eps)
    ratioC = np.abs(Vc) / (np.abs(Ic) + eps)

    df["ratioA"] = ratioA
    df["ratioB"] = ratioB
    df["ratioC"] = ratioC

    df["FaseA"] = 0
    df["FaseB"] = 0
    df["FaseC"] = 0

    return df

df_aug = etiquetar_fisico(df_aug)

# override de etiquetas en ventanas sinteticas
# y etiquetado de falla real C fuera de sintéticos
n = len(df_aug)
inside_synth = np.zeros(n, dtype=bool)

# marcar sinteticos segun eventos
for ev in eventos:
    s, e, fases_ev = ev["start"], ev["end"], ev["phases"]

    # marcar bloque sintetico
    inside_synth[s:e] = True

    # asegurar Falla_Global = 1 en toda la ventana
    df_aug.loc[s:e-1, "Falla_Global"] = 1

    # reset fases y aplicar multietiqueta segun lista de fases
    df_aug.loc[s:e-1, ["FaseA", "FaseB", "FaseC"]] = 0
    for ph in fases_ev:
        df_aug.loc[s:e-1, "Fase" + ph] = 1

# etiquetar la falla real fuera de sinteticos usando fase dominante
ratios = np.vstack([
    df_aug["ratioA"].values,
    df_aug["ratioB"].values,
    df_aug["ratioC"].values
]).T
idx_min = np.argmin(ratios, axis=1)
# 0=A, 1=B, 2=C  
phase_arr = np.array(["A", "B", "C"])

mask_real = (df_aug["Falla_Global"].values == 1) & (~inside_synth)

for i in np.where(mask_real)[0]:
    ph = phase_arr[idx_min[i]]
    df_aug.at[i, "Fase" + ph] = 1

# resumen de distribucion y segmentos
Y = df_aug[["FaseA", "FaseB", "FaseC"]]
print("\nDistribucion conjunta de etiquetas sinteticas y reales:")
print(Y.value_counts())

for col in ["FaseA", "FaseB", "FaseC"]:
    v = df_aug[col].value_counts()
    print(f"{col}: 0={v.get(0,0)}, 1={v.get(1,0)}")

def segmentos_fase(df, col):
    idx = df.index[df[col] == 1].to_numpy()
    if len(idx) == 0:
        return []
    segs = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            segs.append((start, prev))
            start = i
            prev = i
    segs.append((start, prev))
    return segs

for col in ["FaseA", "FaseB", "FaseC"]:
    segs = segmentos_fase(df_aug, col)
    print(f"\nSegmentos detectados para {col}: {len(segs)}")
    for s, e in segs[:5]:
        dt = df_aug["Time"].iloc[1] - df_aug["Time"].iloc[0]
        dur = (e - s + 1) * dt
        print(f"  {s}-{e} (duración ≈ {dur:.4f} s)")

# guardar CSV final para entrenar
df_aug.to_csv("datos_multifase_etiquetado_sintetico.csv", index=False)

# plot evento de ejemplo por fase
def plot_event(df, s, e, titulo):
    t = df["Time"].values
    Ia = df["L4Ia_FLT"].values
    Ib = df["L4Ib_FLT"].values
    Ic = df["L4Ic_FLT"].values
    F_index = df["F_index"].values

    s_plot = max(0, s - 300)
    e_plot = min(len(df) - 1, e + 300)

    plt.figure(figsize=(10, 6))

    # corrientes
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t, Ia, label="Ia")
    ax1.plot(t, Ib, label="Ib")
    ax1.plot(t, Ic, label="Ic")
    ax1.axvspan(t[s], t[e], color="red", alpha=0.2, label="Evento")
    ax1.legend()
    ax1.set_ylabel("I (A)")
    ax1.set_title(titulo)

    # F_index
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(t, F_index)
    ax2.axvspan(t[s], t[e], color="red", alpha=0.2)
    ax2.set_ylabel("F_index")

    # etiquetas
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.step(t, df["FaseA"], label="FaseA")
    ax3.step(t, df["FaseB"], label="FaseB")
    ax3.step(t, df["FaseC"], label="FaseC")
    ax3.axvspan(t[s], t[e], color="red", alpha=0.2)
    ax3.set_ylabel("Etiqueta")
    ax3.set_xlabel("Tiempo (s)")
    ax3.legend()

    plt.xlim(t[s_plot], t[e_plot])
    plt.tight_layout()
    plt.show()

# observar primer segmento de FaseA, B, C 
segs_A = segmentos_fase(df_aug, "FaseA")
if segs_A:
    sA, eA = segs_A[0]
    plot_event(df_aug, sA, eA, "Ejemplo evento Fase A (multi-etiqueta)")

segs_B = segmentos_fase(df_aug, "FaseB")
if segs_B:
    sB, eB = segs_B[0]
    plot_event(df_aug, sB, eB, "Ejemplo evento Fase B (multi-etiqueta)")

segs_C = segmentos_fase(df_aug, "FaseC")
if segs_C:
    sC, eC = segs_C[0]
    plot_event(df_aug, sC, eC, "Ejemplo evento Fase C (multi-etiqueta)")

# Proyecto electrico: Implementación de algoritmos Random Forest y Naive Bayes
# para clasificación multietiqueta en la detección de fallas:
# Desarrollo de una métrica de evaluación comparativa.
# Estudiante: Ebandry Calderón Araya.
# Código que permite extraer la plantilla de falla detectada en la fase C 
# previamente detectada.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# cargar dataset etiquetado
df = pd.read_csv("datos_etiquetados.csv")

# encontrar segmentos Falla_Global = 1
mask = df["Falla_Global"].values.astype(bool)
idx = np.where(mask)[0]

if len(idx) == 0:
    raise ValueError("No hay muestras etiquetadas con Falla_Global = 1")

segments = []
start = idx[0]
prev = idx[0]

for i in idx[1:]:
    if i == prev + 1:
        prev = i
    else:
        segments.append((start, prev))
        start = i
        prev = i
segments.append((start, prev))

print(f"Se encontraron {len(segments)} segmentos de Falla_Global=1:")
for s, e in segments:
    t_ini = df.loc[s, "Time"]
    t_fin = df.loc[e, "Time"]
    dur  = t_fin - t_ini
    print(f"  - Segmento {s:5d}–{e:5d}  "
          f"({e-s+1} muestras, duración ≈ {dur*1000:.2f} ms)")

# se elige segmento mas largo
segments_sorted = sorted(segments, key=lambda x: x[1]-x[0], reverse=True)
s_main, e_main = segments_sorted[0]
t_main_ini = df.loc[s_main, "Time"]
t_main_fin = df.loc[e_main, "Time"]

print("\nSegmento principal de falla:")
print(f"  Índices: {s_main}–{e_main}")
print(f"  Tiempo:  {t_main_ini:.6f} s – {t_main_fin:.6f} s")
print(f"  Duración ≈ {(t_main_fin - t_main_ini)*1000:.2f} ms")

# graficar zona alrededor de la falla
# muestras a cada lado
margen = 500  
i0 = max(0, s_main - margen)
i1 = min(len(df) - 1, e_main + margen)

t  = df["Time"].values[i0:i1+1]
Ia = df["L4Ia_FLT"].values[i0:i1+1]
Ib = df["L4Ib_FLT"].values[i0:i1+1]
Ic = df["L4Ic_FLT"].values[i0:i1+1]
F  = df["F_index"].values[i0:i1+1]
Fg = df["Falla_Global"].values[i0:i1+1]
FA = df["FaseA"].values[i0:i1+1]
FB = df["FaseB"].values[i0:i1+1]
FC = df["FaseC"].values[i0:i1+1]

plt.figure(figsize=(14,8))

# corrientes
plt.subplot(3,1,1)
plt.plot(t, Ia, label="L4Ia", alpha=0.7)
plt.plot(t, Ib, label="L4Ib", alpha=0.7)
plt.plot(t, Ic, label="L4Ic", alpha=0.7)
plt.axvspan(df.loc[s_main,"Time"], df.loc[e_main,"Time"],
            color="red", alpha=0.1, label="Seg. principal")
plt.ylabel("Corriente (A)")
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)

# indice F(t)
plt.subplot(3,1,2)
plt.plot(t, F)
plt.axvspan(df.loc[s_main,"Time"], df.loc[e_main,"Time"],
            color="red", alpha=0.1)
plt.ylabel("F_index = |V|/|I|")
plt.grid(True, alpha=0.3)

# etiquetas
plt.subplot(3,1,3)
plt.step(t, Fg, label="Falla_Global", where="post", color="red")
plt.step(t, FA, label="FaseA", where="post", linestyle="--")
plt.step(t, FB, label="FaseB", where="post", linestyle="--")
plt.step(t, FC, label="FaseC", where="post", linestyle="--")
plt.xlabel("Tiempo (s)")
plt.ylabel("Etiquetas")
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# se extrae plantilla limpia de Fase c
recorte = 10
s_tpl = s_main + recorte
e_tpl = e_main - recorte

plantilla_Ic = df.loc[s_tpl:e_tpl, "L4Ic_FLT"].values.copy()
plantilla_t  = df.loc[s_tpl:e_tpl, "Time"].values.copy()

print(f"\nPlantilla de falla en C extraida: {len(plantilla_Ic)} muestras "
      f"({(plantilla_t[-1]-plantilla_t[0])*1000:.2f} ms utiles)")

np.save("plantilla_Ic_falla_C.npy", plantilla_Ic)
np.save("plantilla_t_falla_C.npy", plantilla_t)

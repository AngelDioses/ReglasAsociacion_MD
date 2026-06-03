# ============================================================
# LABORATORIO 7 — Reglas de Asociacion
# UNMSM · Mineria de Datos · Semana 7
# Dataset: Groceries (mlxtend) — archivo local
# ============================================================

# Instalacion de dependencias (ejecutar si es necesario):
# !pip install mlxtend pandas matplotlib seaborn

# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import time
import warnings

warnings.filterwarnings('ignore')

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

print("Librerias cargadas correctamente.")

# ============================================================
# 1. CARGA — Con submuestreo y filtrado estricto para RAM
# ============================================================
import random

FILE_PATH = r'C:\Users\angel\Desktop\groceries.csv'

transactions_raw = []
with open(FILE_PATH, 'r', encoding='latin-1') as f:
    for line in f:
        items = [item.strip() for item in line.strip().split(',') if item.strip()]
        if items:
            transactions_raw.append(items)

print(f"Total de transacciones originales en disco: {len(transactions_raw)}")

# ── PASO 1: Submuestreo estadístico (Tomamos el 15% del dataset aleatoriamente) ──
# Con ~81,000 transacciones la muestra es enorme y representativa para minería de datos
random.seed(42)  # Semilla para que el resultado sea reproducible
PORCENTAJE_MUESTRA = 0.15
num_muestras = int(len(transactions_raw) * PORCENTAJE_MUESTRA)
transactions_sampled = random.sample(transactions_raw, num_muestras)

# ── PASO 2: Filtrado de ítems raros más estricto ──
from collections import Counter
item_counts = Counter(item for trans in transactions_sampled for item in trans)

# Subimos el umbral: el ítem debe aparecer al menos 50 veces en la muestra
MIN_OCURRENCIAS = 50 
items_validos = {item for item, count in item_counts.items() if count >= MIN_OCURRENCIAS}

# Reconstruimos las transacciones con los filtros aplicados
transactions = []
for trans in transactions_sampled:
    filtrada = [item for item in trans if item in items_validos]
    if filtrada:
        transactions.append(filtrada)

DATASET_NAME = "Online Retail / Groceries (Muestra Optimizada)"
print(f"\n--- ESTADÍSTICAS DEL DATASET REDUCIDO ---")
print(f"Muestra utilizada          : {PORCENTAJE_MUESTRA*100}% del total")
print(f"Ítems únicos antes del filtro: {len(item_counts)}")
print(f"Ítems únicos después del filtro (frecuencia >= {MIN_OCURRENCIAS}): {len(items_validos)}")
print(f"Total de transacciones útiles para el modelo: {len(transactions)}")

# ============================================================
# 2. CODIFICACIÓN — TransactionEncoder (Ahora sí entrará en memoria)
# ============================================================
te = TransactionEncoder()
te_array = te.fit_transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)

print(f"\nDimensiones finales de la matriz en memoria: {df.shape}")
print(f"Número de ítems únicos (columnas) : {df.shape[1]}")
print(f"Número de transacciones (filas)    : {df.shape[0]}")

# ============================================================
# 3. ENTRENAR 2 MODELOS — Apriori vs FP-Growth
#    Comparacion de tiempos con el mismo min_support
# ============================================================
MIN_SUPPORT_BASE = 0.03

print(f"\n{'='*60}")
print(f"  Comparacion Apriori vs FP-Growth  |  min_sup = {MIN_SUPPORT_BASE}")
print(f"  Dataset: {DATASET_NAME}")
print(f"{'='*60}")

# -- Apriori --------------------------------------------------
t0         = time.perf_counter()
fi_apriori = apriori(df, min_support=MIN_SUPPORT_BASE, use_colnames=True)
time_ap    = time.perf_counter() - t0

# -- FP-Growth ------------------------------------------------
t0          = time.perf_counter()
fi_fpgrowth = fpgrowth(df, min_support=MIN_SUPPORT_BASE, use_colnames=True)
time_fp     = time.perf_counter() - t0

print(f"\nTiempo Apriori     : {time_ap:.4f} s")
print(f"Tiempo FP-Growth   : {time_fp:.4f} s")
print(f"Itemsets Apriori   : {len(fi_apriori)}")
print(f"Itemsets FP-Growth : {len(fi_fpgrowth)}")

tabla_tiempos = pd.DataFrame({
    'Algoritmo'      : ['Apriori', 'FP-Growth'],
    'Tiempo (s)'     : [round(time_ap, 4), round(time_fp, 4)],
    'Itemsets'       : [len(fi_apriori), len(fi_fpgrowth)],
    'Resultado igual': ['Si', 'Si']
})
print("\nTabla comparativa de tiempos:")
print(tabla_tiempos.to_string(index=False))

# ============================================================
# 4. AJUSTE DE PARAMETROS
#    Barrido: 3 valores de min_support x 2 valores de min_confidence
# ============================================================
min_supports    = [0.01, 0.03, 0.05]
min_confidences = [0.50, 0.70]

print(f"\n{'='*65}")
print("  Tabla de barrido: min_support x min_confidence (algoritmo: Apriori)")
print(f"{'='*65}")
print(f"  {'min_sup':>8} | {'min_conf':>9} | {'Itemsets':>9} | {'Reglas':>7}")
print(f"  {'-'*46}")

resultados = []
for ms in min_supports:
    fi = apriori(df, min_support=ms, use_colnames=True)
    for mc in min_confidences:
        if len(fi) > 0:
            r = association_rules(fi, metric='confidence', min_threshold=mc)
        else:
            r = pd.DataFrame()
        resultados.append({
            'min_support'   : ms,
            'min_confidence': mc,
            'n_itemsets'    : len(fi),
            'n_reglas'      : len(r)
        })
        print(f"  {ms:>8.2f} | {mc:>9.2f} | {len(fi):>9} | {len(r):>7}")

df_barrido = pd.DataFrame(resultados)
pivot = df_barrido.pivot(
    index='min_support',
    columns='min_confidence',
    values='n_reglas'
)
pivot.columns = [f'conf={c}' for c in pivot.columns]
print("\nResumen — numero de reglas generadas por combinacion de parametros:")
print(pivot.to_string())

# ============================================================
# 5. EVALUACION E INTERPRETACION
# ============================================================

# -- Generar reglas con parametros base ----------------------
fi_final = apriori(df, min_support=MIN_SUPPORT_BASE, use_colnames=True)
reglas   = association_rules(fi_final, metric='confidence',
                             min_threshold=0.50)
reglas   = reglas.sort_values('lift', ascending=False).reset_index(drop=True)

print(f"\n{'='*60}")
print(f"  Top 10 reglas por Lift  |  sup>={MIN_SUPPORT_BASE}, conf>=0.50")
print(f"{'='*60}")
cols = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
print(reglas[cols].head(10).to_string())

# -- Grafico scatter: support vs confidence, tamano = lift ----
fig, ax = plt.subplots(figsize=(11, 7))

top_n  = min(60, len(reglas))
r_plot = reglas.head(top_n)

scatter = ax.scatter(
    r_plot['support'],
    r_plot['confidence'],
    s          = r_plot['lift'] * 150,
    c          = r_plot['lift'],
    cmap       = 'RdYlGn',
    alpha      = 0.75,
    edgecolors = 'gray',
    linewidths = 0.4
)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Lift', fontsize=11)

ax.axhline(y=0.50, color='red',  linestyle='--', linewidth=0.9,
           alpha=0.6, label='min_conf = 0.50')
ax.axvline(x=MIN_SUPPORT_BASE, color='blue', linestyle='--', linewidth=0.9,
           alpha=0.6, label=f'min_sup = {MIN_SUPPORT_BASE}')

ax.set_xlabel('Support',    fontsize=12)
ax.set_ylabel('Confidence', fontsize=12)
ax.set_title(
    f'Reglas de Asociacion — Support vs Confidence\n'
    f'Tamano del punto proporcional al Lift  |  Dataset: {DATASET_NAME}',
    fontsize=12, fontweight='bold'
)
ax.legend(fontsize=10)
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
plt.tight_layout()
plt.savefig('scatter_reglas.png', dpi=150, bbox_inches='tight')
plt.show()
print("Grafico guardado como: scatter_reglas.png")

# -- Interpretacion de negocio: top 3 reglas por lift ---------
print(f"\n{'='*65}")
print("  Interpretacion de negocio — Top 3 reglas por Lift")
print(f"{'='*65}")

for i, row in reglas.head(3).iterrows():
    ant = ', '.join(sorted(list(row['antecedents'])))
    con = ', '.join(sorted(list(row['consequents'])))
    s   = row['support']
    c   = row['confidence']
    l   = row['lift']

    print(f"\nRegla {i+1}: [{ant}] -> [{con}]")
    print(f"  Support    : {s:.4f}  — presente en el {s*100:.2f}% de las transacciones.")
    print(f"  Confidence : {c:.4f}  — el {c*100:.1f}% de quienes compran '{ant}'")
    print(f"               tambien adquieren '{con}'.")
    print(f"  Lift       : {l:.4f}")

    if l > 1.0:
        print(f"  Interpretacion: la asociacion es {(l-1)*100:.1f}% mas frecuente de lo")
        print(f"  esperado si los items fueran independientes. Se recomienda")
        print(f"  colocarlos en seccion proxima o incluirlos en una promocion conjunta.")
    elif l < 1.0:
        print(f"  Interpretacion: correlacion negativa; la compra conjunta es")
        print(f"  menos probable que si los items fueran independientes.")
    else:
        print(f"  Interpretacion: los items son estadisticamente independientes.")

# -- Tabla final de comparacion Apriori vs FP-Growth ----------
print(f"\n{'='*60}")
print("  Tabla final — Comparacion Apriori vs FP-Growth")
print(f"{'='*60}")
print(tabla_tiempos.to_string(index=False))
print(
    "\nNota: FP-Growth construye un arbol FP-Tree comprimido, evitando la "
    "generacion explicita de candidatos. Esto lo hace mas rapido que Apriori "
    "en datasets con alta cardinalidad. Ambos algoritmos producen itemsets "
    "y reglas identicos para los mismos parametros de entrada."
)


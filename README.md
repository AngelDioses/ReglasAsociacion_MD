# Laboratorio 7 — Reglas de Asociación

## Requisitos

```bash
pip install mlxtend pandas numpy matplotlib
```

## Dataset

Descargar `groceries.csv` desde el repositorio en Kaggle:  
[Dataset](https://www.kaggle.com/datasets/vijayuv/onlineretail?resource=download)

Guardar el archivo como `groceries.csv` en el entorno de trabajo.

## Ejecución en Google Colab

1. Subir `groceries.csv` al entorno (panel lateral → subir archivo).
2. Copiar el código .py a su entorno en Jupyter.
3. Ejecutar todas las celdas: **Runtime → Run all**.

## Ejecución en Jupyter local

```bash
pip install mlxtend pandas numpy matplotlib
jupyter notebook lab7_asociacion.ipynb
```

Ejecutar todas las celdas: **Kernel → Restart & Run All**.

## Parámetros

```python
FILE_PATH        = '/content/groceries.csv'  # Ajustar según entorno
MIN_SUPPORT_BASE = 0.03
min_supports     = [0.01, 0.03, 0.05]
min_confidences  = [0.50, 0.70]
```

## Archivos generados

- `scatter_reglas.png` — Gráfico scatter de las reglas de asociación.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# =========================================================
# 1) Lectura del primer dataset
# =========================================================
df = pd.read_csv('countries1.csv', sep=";")
print(df.head(5))

print('Cantidad de filas y columnas:', df.shape)
print('Nombres de columnas:', df.columns.tolist())
df.info()
print(df.describe())   # FIX: faltaba print() — sin él, en un .py el resultado se descarta

# =========================================================
# 2) Matriz de correlación
# =========================================================
numeric_df = df.select_dtypes(include=[np.number])
corre = numeric_df.corr()
sm.graphics.plot_corr(corre, xnames=list(corre.columns))
plt.show()

# =========================================================
# 3) Lectura del segundo dataset (población histórica)
# =========================================================
df_pop = pd.read_csv('countries2.csv')
print(df_pop.head(5))

# --- España ---
df_pop_es = df_pop[df_pop["country"] == 'Spain']
print(df_pop_es.head())

# FIX: el original hacía .drop(['country'], axis=1)['population'].plot(...),
#      lo que mostraba los números de fila (1416, 1417...) en el eje X
#      en lugar de los años. Lo correcto es usar 'year' como índice.
df_pop_es.set_index('year')['population'].plot(kind='bar')
plt.title('Población de España por año')
plt.show()

# --- Argentina + comparativa ---
df_pop_ar = df_pop[df_pop["country"] == 'Argentina']
print(df_pop_ar.head())

anios = df_pop_es['year'].unique()
pop_ar = df_pop_ar['population'].values
pop_es = df_pop_es['population'].values

df_plot = pd.DataFrame({'Argentina': pop_ar, 'Spain': pop_es}, index=anios)
df_plot.plot(kind='bar')
plt.title('Población Argentina vs España')
plt.show()

# =========================================================
# 4) Filtrado de países hispanohablantes
# =========================================================
# FIX 1: regex=True es innecesario para reemplazar NaN por '' y en pandas
#        recientes genera FutureWarning. Se pone regex=False (o se omite).
# FIX 2: .str.contains('es') como substring es frágil — podría matchear
#        códigos como 'ces' (checo). Se filtra dividiendo por coma y
#        comparando contra 'es' o prefijo 'es-' (es-AR, es-MX, etc.).
# FIX 3: .copy() para evitar SettingWithCopyWarning más adelante.
df_espanol = df.replace(np.nan, '', regex=False).copy()
mask = df_espanol['languages'].str.split(',').apply(
    lambda langs: any(l.strip() == 'es' or l.strip().startswith('es-') for l in langs)
)
df_espanol = df_espanol[mask].copy()
print(df_espanol)

df_espanol.set_index('alpha_3')[['population', 'area']].plot(
    kind='bar', rot=65, figsize=(20, 10)
)
plt.show()

# =========================================================
# 5) Detección de anomalías (outliers por desviación estándar)
# =========================================================
# FIX 1: la lista `anomalies` era GLOBAL y se acumulaba entre llamadas.
#        Si la función se llamaba dos veces, devolvía duplicados.
# FIX 2: el valor de retorno se ignoraba y luego se hacían .drop([30, 233])
#        con índices hardcodeados — frágil ante cualquier cambio de datos.
# FIX 3: se simplifica usando la Serie directamente en vez de iterrows().
def find_anomalies(data):
    anomalies = []                          # local — no estado compartido
    col = data.columns[0]
    serie = data[col]
    cutoff = serie.std() * 2
    lower = serie.mean() - cutoff
    upper = serie.mean() + cutoff
    print(f'Límite inferior: {lower:.2f}')
    print(f'Límite superior: {upper:.2f}')
    for idx, val in serie.items():
        if val > upper or val < lower:
            anomalies.append(idx)
    return anomalies

# Indexamos por alpha_3 para que las anomalías sean etiquetas legibles
df_espanol_idx = df_espanol.set_index('alpha_3')
outliers = find_anomalies(df_espanol_idx[['population']])
print('Anomalías detectadas:', outliers)

# FIX 4: usamos el resultado de la función y .drop por etiqueta,
#        en vez de los índices posicionales hardcodeados [30, 233].
df_espanol_filtrado = df_espanol_idx.drop(outliers)
df_espanol_filtrado[['population', 'area']] \
    .sort_values('population') \
    .plot(kind='bar', rot=65, figsize=(20, 10))
plt.show()

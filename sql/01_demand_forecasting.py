# %% [markdown]
# # Demand & Inventory Forecasting
# **Objetivo:** Predecir la demanda mensual por producto y almacén para optimizar niveles de inventario.
#
# **Stack:** Python, pandas, scikit-learn, matplotlib
#
# **Datos:** Histórico simulado de ventas e inventario 2022-2023 (refacciones automotrices)

# %% [markdown]
# ## 1. Setup e importaciones

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["font.size"] = 11

import os
os.makedirs("../img", exist_ok=True)

# %% [markdown]
# ## 2. Carga de datos

# %%
sales = pd.read_csv("../data/sales_raw.csv", parse_dates=["date"])
inventory = pd.read_csv("../data/inventory_raw.csv", parse_dates=["date"])

print("Sales shape:", sales.shape)
print("Inventory shape:", inventory.shape)
sales.head()

# %%
inventory.head()

# %% [markdown]
# ## 3. Limpieza y validación

# %%
# Verificar nulos
print("Nulos en sales:")
print(sales.isnull().sum())
print("\nNulos en inventory:")
print(inventory.isnull().sum())

# %%
# Verificar tipos
print(sales.dtypes)

# %%
# Verificar rangos
print("Units sold - min:", sales["units_sold"].min(), "max:", sales["units_sold"].max())
print("Revenue - min:", sales["revenue"].min(), "max:", sales["revenue"].max())
print("Stock final negativo:", (inventory["stock_final"] < 0).sum())

# %%
# Registros duplicados
print("Duplicados en sales:", sales.duplicated().sum())
print("Duplicados en inventory:", inventory.duplicated().sum())

# %% [markdown]
# ## 4. EDA - Análisis Exploratorio

# %% [markdown]
# ### 4.1 Demanda mensual total

# %%
monthly_total = (
    sales.groupby("date")["units_sold"]
    .sum()
    .reset_index()
    .rename(columns={"date": "month"})
)

fig, ax = plt.subplots()
ax.plot(monthly_total["month"], monthly_total["units_sold"], marker="o", linewidth=2, color="#2563EB")
ax.set_title("Demanda mensual total - todos los productos y almacenes")
ax.set_xlabel("Mes")
ax.set_ylabel("Unidades vendidas")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../img/01_demanda_mensual_total.png", dpi=150)
plt.show()

# %% [markdown]
# ### 4.2 Demanda por producto

# %%
by_product = (
    sales.groupby(["date", "product_name"])["units_sold"]
    .sum()
    .reset_index()
)

products = by_product["product_name"].unique()
fig, axes = plt.subplots(4, 2, figsize=(14, 16))
axes = axes.flatten()

for i, product in enumerate(products):
    data = by_product[by_product["product_name"] == product]
    axes[i].plot(data["date"], data["units_sold"], marker="o", linewidth=1.5, color="#7C3AED")
    axes[i].set_title(product, fontsize=10)
    axes[i].set_xlabel("")
    axes[i].tick_params(axis="x", rotation=45)

plt.suptitle("Demanda mensual por producto", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("../img/02_demanda_por_producto.png", dpi=150)
plt.show()

# %% [markdown]
# ### 4.3 Revenue por almacén

# %%
by_warehouse = (
    sales.groupby(["date", "warehouse"])["revenue"]
    .sum()
    .reset_index()
)

fig, ax = plt.subplots()
colors = {"CDMX": "#2563EB", "GDL": "#16A34A", "MTY": "#DC2626"}

for warehouse, group in by_warehouse.groupby("warehouse"):
    ax.plot(group["date"], group["revenue"], marker="o", linewidth=2,
            label=warehouse, color=colors[warehouse])

ax.set_title("Revenue mensual por almacén")
ax.set_xlabel("Mes")
ax.set_ylabel("Revenue (MXN)")
ax.legend()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../img/03_revenue_por_almacen.png", dpi=150)
plt.show()

# %% [markdown]
# ### 4.4 Análisis de stockouts

# %%
stockout_summary = (
    inventory.groupby(["product_name", "warehouse"])
    .agg(
        total_months=("stockout", "count"),
        months_with_stockout=("stockout", "sum")
    )
    .assign(stockout_rate=lambda df: df["months_with_stockout"] / df["total_months"] * 100)
    .reset_index()
    .sort_values("stockout_rate", ascending=False)
)

print(stockout_summary.head(10).to_string(index=False))

# %%
# Top 10 combinaciones producto/almacén con mayor riesgo
top10 = stockout_summary.head(10).copy()
top10["label"] = top10["product_name"].str[:12] + " / " + top10["warehouse"]

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.barh(top10["label"], top10["stockout_rate"], color="#DC2626", alpha=0.8)
ax.set_xlabel("Stockout rate (%)")
ax.set_title("Top 10 combinaciones producto/almacén con mayor tasa de rotura de stock")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("../img/04_stockout_rate.png", dpi=150)
plt.show()

# %% [markdown]
# ## 5. Feature Engineering

# %%
# Agregar ventas por producto + almacén + mes (nivel de granularidad del modelo)
df = (
    sales.groupby(["date", "product_id", "product_name", "category", "warehouse"])
    .agg(units_sold=("units_sold", "sum"), revenue=("revenue", "sum"))
    .reset_index()
    .sort_values(["product_id", "warehouse", "date"])
)

df.head()

# %%
# Features temporales
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["quarter"] = df["date"].dt.quarter

# Lag features por producto + almacén
df = df.sort_values(["product_id", "warehouse", "date"])

for lag in [1, 2, 3]:
    df[f"lag_{lag}"] = df.groupby(["product_id", "warehouse"])["units_sold"].shift(lag)

# Rolling mean (3 meses)
df["rolling_mean_3"] = (
    df.groupby(["product_id", "warehouse"])["units_sold"]
    .transform(lambda x: x.shift(1).rolling(3).mean())
)

# Encoding de variables categóricas
le_product = LabelEncoder()
le_warehouse = LabelEncoder()
le_category = LabelEncoder()

df["product_enc"] = le_product.fit_transform(df["product_id"])
df["warehouse_enc"] = le_warehouse.fit_transform(df["warehouse"])
df["category_enc"] = le_category.fit_transform(df["category"])

# Eliminar filas con NaN (generadas por lags)
df_model = df.dropna().copy()

print("Shape para modelado:", df_model.shape)
df_model.head()

# %% [markdown]
# ## 6. Modelo de Forecasting - Random Forest

# %%
FEATURES = [
    "year", "month", "quarter",
    "lag_1", "lag_2", "lag_3",
    "rolling_mean_3",
    "product_enc", "warehouse_enc", "category_enc"
]
TARGET = "units_sold"

X = df_model[FEATURES]
y = df_model[TARGET]

# Time Series Split para validación
tscv = TimeSeriesSplit(n_splits=3)

mae_scores = []
mape_scores = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    mae_scores.append(mae)
    mape_scores.append(mape)
    print(f"Fold {fold+1} | MAE: {mae:.1f} | MAPE: {mape:.1f}%")

print(f"\nMAE promedio:  {np.mean(mae_scores):.1f}")
print(f"MAPE promedio: {np.mean(mape_scores):.1f}%")

# %%
# Entrenar modelo final con todos los datos
model_final = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)
model_final.fit(X, y)

# Feature importance
importance_df = (
    pd.DataFrame({"feature": FEATURES, "importance": model_final.feature_importances_})
    .sort_values("importance", ascending=False)
)

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(importance_df["feature"], importance_df["importance"], color="#2563EB", alpha=0.8)
ax.set_title("Feature Importance - Random Forest")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("../img/05_feature_importance.png", dpi=150)
plt.show()

# %% [markdown]
# ## 7. Visualización: Real vs Predicho

# %%
# Predicción sobre el set completo (para visualización)
df_model = df_model.copy()
df_model["predicted"] = model_final.predict(X)

# Seleccionar un producto/almacén para graficar
sample = df_model[
    (df_model["product_name"] == "Filtro de aceite") &
    (df_model["warehouse"] == "GDL")
].sort_values("date")

fig, ax = plt.subplots()
ax.plot(sample["date"], sample["units_sold"], marker="o", label="Real", color="#2563EB", linewidth=2)
ax.plot(sample["date"], sample["predicted"], marker="s", label="Predicho", color="#DC2626",
        linewidth=2, linestyle="--")
ax.set_title("Demanda real vs predicha - Filtro de aceite / GDL")
ax.set_xlabel("Mes")
ax.set_ylabel("Unidades vendidas")
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../img/06_real_vs_predicho.png", dpi=150)
plt.show()

# %% [markdown]
# ## 8. Business Layer: Tabla de reposición sugerida

# %%
# Unir predicciones con inventario actual (último mes disponible)
last_month = inventory["date"].max()
inv_last = inventory[inventory["date"] == last_month][
    ["product_id", "product_name", "warehouse", "stock_final", "reorder_point", "lead_time_days"]
].copy()

# Forecast del siguiente mes: usar la media de predicciones del último trimestre por producto/almacén
forecast_next = (
    df_model[df_model["date"] >= df_model["date"].max() - pd.DateOffset(months=2)]
    .groupby(["product_id", "warehouse"])["predicted"]
    .mean()
    .round()
    .reset_index()
    .rename(columns={"predicted": "forecast_next_month"})
)

business_table = inv_last.merge(forecast_next, on=["product_id", "warehouse"], how="left")

# Unidades sugeridas a pedir
business_table["units_to_order"] = (
    business_table["forecast_next_month"] - business_table["stock_final"]
).clip(lower=0).round()

business_table["action"] = business_table["units_to_order"].apply(
    lambda x: "REORDER" if x > 0 else "OK"
)

business_table = business_table.sort_values("units_to_order", ascending=False)

print(business_table.to_string(index=False))
business_table.to_csv("../data/reorder_suggestions.csv", index=False)
print("\nTabla exportada: data/reorder_suggestions.csv")

# %% [markdown]
# ## 9. Resumen de resultados
#
# | Métrica | Valor |
# |---|---|
# | Modelo | Random Forest Regressor |
# | Granularidad | Producto x Almacén x Mes |
# | MAE promedio (CV) | ver output arriba |
# | MAPE promedio (CV) | ver output arriba |
# | Features clave | lag_1, rolling_mean_3, month |
# | Output de negocio | Tabla de reposición sugerida por producto/almacén |
#
# **Siguiente paso:** Integrar el forecast con alertas automáticas en el ERP/WMS.
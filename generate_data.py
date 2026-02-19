"""
generate_data.py
Genera datasets simulados para el proyecto demand-inventory-forecasting.
Archivos de salida: data/sales_raw.csv, data/inventory_raw.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)
Path("data").mkdir(exist_ok=True)

# --- Configuración base ---
PRODUCTS = {
    "P001": "Filtro de aceite",
    "P002": "Pastillas de freno",
    "P003": "Bujías",
    "P004": "Amortiguador trasero",
    "P005": "Correa de distribución",
    "P006": "Radiador",
    "P007": "Batería 12V",
    "P008": "Faro delantero",
}

CATEGORIES = {
    "P001": "Mantenimiento",
    "P002": "Frenos",
    "P003": "Encendido",
    "P004": "Suspensión",
    "P005": "Motor",
    "P006": "Refrigeración",
    "P007": "Eléctrico",
    "P008": "Iluminación",
}

WAREHOUSES = ["CDMX", "GDL", "MTY"]

# Fechas: 2 años de histórico mensual (2022-2023)
dates = pd.date_range(start="2022-01-01", end="2023-12-01", freq="MS")

# -------------------------------------------------------
# SALES_RAW.CSV
# -------------------------------------------------------
records = []

for product_id, product_name in PRODUCTS.items():
    base_demand = np.random.randint(80, 400)

    for date in dates:
        for warehouse in WAREHOUSES:
            # Tendencia ligera al alza
            trend = (date.year - 2022) * 12 + date.month
            trend_factor = 1 + trend * 0.004

            # Estacionalidad: pico en enero/julio (talleres), baja en diciembre
            seasonality = {
                1: 1.20, 2: 1.05, 3: 1.00, 4: 0.95, 5: 0.98,
                6: 1.02, 7: 1.18, 8: 1.05, 9: 1.00, 10: 0.97,
                11: 0.95, 12: 0.85
            }[date.month]

            # Variación por almacén
            warehouse_factor = {"CDMX": 1.30, "GDL": 1.00, "MTY": 0.85}[warehouse]

            # Ruido aleatorio
            noise = np.random.normal(1.0, 0.08)

            units_sold = max(
                0,
                int(base_demand * trend_factor * seasonality * warehouse_factor * noise)
            )

            unit_price = round(np.random.uniform(120, 2500), 2)

            records.append({
                "date": date.strftime("%Y-%m-%d"),
                "product_id": product_id,
                "product_name": product_name,
                "category": CATEGORIES[product_id],
                "warehouse": warehouse,
                "units_sold": units_sold,
                "unit_price": unit_price,
                "revenue": round(units_sold * unit_price, 2),
            })

sales_df = pd.DataFrame(records)
sales_df.to_csv("data/sales_raw.csv", index=False)
print(f"sales_raw.csv generado: {len(sales_df)} registros")

# -------------------------------------------------------
# INVENTORY_RAW.CSV
# -------------------------------------------------------
inv_records = []

for product_id, product_name in PRODUCTS.items():
    for date in dates:
        for warehouse in WAREHOUSES:
            # Filtrar ventas del mes para ese producto/almacén
            sold = sales_df[
                (sales_df["product_id"] == product_id) &
                (sales_df["warehouse"] == warehouse) &
                (sales_df["date"] == date.strftime("%Y-%m-%d"))
            ]["units_sold"].values[0]

            stock_inicial = int(sold * np.random.uniform(0.6, 1.4))
            replenishment = int(sold * np.random.uniform(0.3, 0.9))  # reposición incompleta a veces
            stock_final = max(0, stock_inicial - sold + replenishment)
            reorder_point = int(sold * 0.6)
            lead_time_days = np.random.randint(3, 15)

            inv_records.append({
                "date": date.strftime("%Y-%m-%d"),
                "product_id": product_id,
                "product_name": product_name,
                "category": CATEGORIES[product_id],
                "warehouse": warehouse,
                "stock_inicial": stock_inicial,
                "units_sold": sold,
                "stock_final": stock_final,
                "reorder_point": reorder_point,
                "lead_time_days": lead_time_days,
                "stockout": int(stock_final == 0),
            })

inventory_df = pd.DataFrame(inv_records)
inventory_df.to_csv("data/inventory_raw.csv", index=False)
print(f"inventory_raw.csv generado: {len(inventory_df)} registros")
print("\nListo. Archivos en /data/")
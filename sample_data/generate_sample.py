"""Creates sample_data/sales_data.csv (realistic fake sales data, with a few missing values and outliers)."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1000
categories = {"Electronics": 1.6, "Furniture": 1.2, "Clothing": 0.6, "Groceries": 0.4}
regions = ["North", "South", "East", "West"]
cat = rng.choice(list(categories), n, p=[0.3, 0.2, 0.3, 0.2])
dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
qty = rng.integers(1, 10, n)
base = np.array([categories[c] for c in cat]) * 1000
sales = (base * qty * rng.uniform(0.5, 1.2, n) / 3).round(2)
sales[dates.month >= 10] *= 1.3          # Q4 boost
sales[rng.choice(n, 6, replace=False)] *= 8   # outliers
profit = (sales * rng.uniform(0.05, 0.3, n)).round(2)
df = pd.DataFrame({
    "Order_ID": range(1001, 1001 + n),
    "Order_Date": dates.strftime("%Y-%m-%d"),
    "Customer": [f"Customer {i}" for i in rng.integers(1, 60, n)],
    "Category": cat,
    "Region": rng.choice(regions, n),
    "Quantity": qty,
    "Sales": sales.round(2),
    "Profit": profit,
})
df.loc[rng.choice(n, 25, replace=False), "Region"] = np.nan
df.loc[rng.choice(n, 15, replace=False), "Profit"] = np.nan
df.sort_values("Order_Date").to_csv("sample_data/sales_data.csv", index=False)
print("Created sample_data/sales_data.csv")

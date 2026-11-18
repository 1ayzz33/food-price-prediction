"""Generate a realistic Nigerian food price dataset for the ML project.

Produces ~300 rows across 8 commodities, 12 states, and 12 months with
realistic relationships between fuel price, inflation, rainfall and market price.
"""
from __future__ import annotations

import random
from datetime import date

import pandas as pd

random.seed(42)

COMMODITIES = ["Rice", "Beans", "Maize", "Millet", "Sorghum", "Yam", "Cassava", "Wheat"]
STATES = [
    "Kano", "Kaduna", "Katsina", "Sokoto", "Borno", "Adamawa", "Bauchi", "Yobe",
    "Zamfara", "Jigawa", "Kebbi", "Niger",
]

# Base price range per commodity (₦ per unit, e.g. per 50kg bag equivalent)
BASE_PRICE = {
    "Rice": (85000, 145000),
    "Beans": (95000, 175000),
    "Maize": (50000, 100000),
    "Millet": (60000, 110000),
    "Sorghum": (55000, 105000),
    "Yam": (110000, 200000),
    "Cassava": (40000, 90000),
    "Wheat": (100000, 190000),
}

# State price multiplier (production states = lower, urban/consumption = higher)
STATE_MULTIPLIER = {
    "Kano": 0.95, "Kaduna": 1.0, "Katsina": 0.92, "Sokoto": 0.98,
    "Borno": 1.05, "Adamawa": 1.02, "Bauchi": 0.96, "Yobe": 1.03,
    "Zamfara": 0.94, "Jigawa": 0.93, "Kebbi": 0.91, "Niger": 0.89,
}

# Month effect: planting/harvest seasons affect supply and price
MONTH_EFFECT = {
    1: 1.02, 2: 1.01, 3: 1.00, 4: 0.98, 5: 0.96, 6: 0.95,
    7: 0.97, 8: 0.99, 9: 1.01, 10: 1.03, 11: 1.05, 12: 1.04,
}

rows: list[dict] = []

for _ in range(300):
    commodity = random.choice(COMMODITIES)
    state = random.choice(STATES)
    month = random.randint(1, 12)

    fuel_price = round(random.uniform(800, 1200), 0)
    inflation = round(random.uniform(20, 40), 1)
    rainfall = round(random.uniform(0, 300), 0)

    # Base price with commodity and state adjustments
    low, high = BASE_PRICE[commodity]
    base = random.uniform(low, high)

    # Fuel price effect: every ₦100 above 900 adds ~1.5% to price
    fuel_factor = 1 + (fuel_price - 900) * 0.000015

    # Inflation effect: every 1% above 28 adds ~0.8% to price
    inflation_factor = 1 + (inflation - 28) * 0.008

    # Rainfall effect: moderate rainfall (100-200mm) lowers price, extremes raise it
    if 100 <= rainfall <= 200:
        rain_factor = 0.97  # good growing conditions
    elif rainfall < 50:
        rain_factor = 1.06  # drought
    elif rainfall > 250:
        rain_factor = 1.04  # flooding / transport disruption
    else:
        rain_factor = 1.0

    price = (
        base
        * STATE_MULTIPLIER[state]
        * MONTH_EFFECT[month]
        * fuel_factor
        * inflation_factor
        * rain_factor
    )
    price = round(price, -2)  # round to nearest ₦100

    rows.append(
        {
            "Commodity": commodity,
            "State": state,
            "Month": month,
            "Fuel Price": int(fuel_price),
            "Inflation": inflation,
            "Rainfall": int(rainfall),
            "Price": int(price),
        }
    )

df = pd.DataFrame(rows)
df.to_csv("food_price.csv", index=False)
print(f"Generated {len(df)} rows across {df['Commodity'].nunique()} commodities "
      f"and {df['State'].nunique()} states.")
print(df.head())
print(df["Price"].describe())

"""
Predict temperature at power unit locations using kriging.

Predicts: temp_diff, plus min/max/avg temp for 2023 & 2024.
Output: one Excel sheet per variable, with date-range column headers.

Requirements:
  pip install pandas numpy pyproj pykrige openpyxl python-calamine
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging
from pyproj import Transformer

# --- Input files ---
TEMP_DIFF_PATH = Path(
    "/Users/jackey/Desktop/Kriging/"
    "Corrected_City-Specific_Weekly_Temperature_Differences_with_Location.xlsx"
)
WEEKLY_TEMP_PATH = Path(
    "/Users/jackey/Desktop/Kriging/china_summer_temperatures_weekly.xlsx"
)
GEOCODED_PATH = Path(
    "/Users/jackey/Desktop/Chinese_Temp_Predictions/geocoded_city_districts.xlsx"
)

# --- Output ---
OUTPUT_DIR = Path("/Users/jackey/Desktop/Chinese_Temp_Predictions")
CITY_OUTPUT = OUTPUT_DIR / "city_temp_predictions.xlsx"
DISTRICT_OUTPUT = OUTPUT_DIR / "district_temp_predictions.xlsx"

# Variables to krige from the weekly temperature dataset, per year
WEEKLY_VARS = ["min_temp", "max_temp", "avg_temp"]
YEARS = [2023, 2024]


def load_power_unit_coords() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load geocoded city and district coordinates."""
    cities = pd.read_excel(GEOCODED_PATH, sheet_name="Unique_Cities")
    districts = pd.read_excel(GEOCODED_PATH, sheet_name="Unique_Districts")

    cities = cities[cities["latitude"].notna()].copy()
    districts = districts[districts["latitude"].notna()].copy()

    print(f"Power units: {len(cities)} cities, {len(districts)} districts with coordinates")
    return cities, districts


def run_kriging(
    station_lons: np.ndarray,
    station_lats: np.ndarray,
    station_values: np.ndarray,
    target_lons: np.ndarray,
    target_lats: np.ndarray,
    transformer: Transformer,
) -> np.ndarray:
    """Run Ordinary Kriging and predict at target locations."""
    sx, sy = transformer.transform(station_lons, station_lats)
    tx, ty = transformer.transform(target_lons, target_lats)

    ok = OrdinaryKriging(
        sx, sy, station_values,
        variogram_model="spherical",
        verbose=False,
        enable_plotting=False,
    )

    z, _ = ok.execute("points", tx, ty)
    return np.asarray(z)


def predict_variable(
    weather_df: pd.DataFrame,
    value_col: str,
    target_lons: np.ndarray,
    target_lats: np.ndarray,
    transformer: Transformer,
    label: str = "",
) -> dict[int, np.ndarray]:
    """Predict a single variable at target locations for each week.

    Returns {week_number: predictions_array}.
    """
    weeks = sorted(weather_df["week_number"].unique())
    results: dict[int, np.ndarray] = {}

    for week in weeks:
        week_data = weather_df[weather_df["week_number"] == week].copy()
        week_data = week_data.dropna(subset=[value_col])
        week_data = week_data.drop_duplicates(subset=["longitude", "latitude"])

        if len(week_data) < 5:
            print(f"  {label} week {week}: skipped ({len(week_data)} stations)")
            continue

        try:
            preds = run_kriging(
                week_data["longitude"].to_numpy(),
                week_data["latitude"].to_numpy(),
                week_data[value_col].to_numpy(),
                target_lons,
                target_lats,
                transformer,
            )
            results[week] = preds
            print(f"  {label} week {week}: OK ({len(week_data)} stations)")
        except Exception as e:
            print(f"  {label} week {week}: FAILED - {e}")

    return results


# Week number -> date range label (year-agnostic, using Mon-Day format)
WEEK_LABELS = {
    22: "Jun 03-09",
    23: "Jun 10-16",
    24: "Jun 17-23",
    25: "Jun 24-30",
    26: "Jul 01-07",
    27: "Jul 08-14",
    28: "Jul 15-21",
    29: "Jul 22-28",
    30: "Jul 29-Aug 04",
    31: "Aug 05-11",
    32: "Aug 12-18",
    33: "Aug 19-25",
    34: "Aug 26-Sep 01",
    35: "Aug 26-Sep 01",  # 2024 week 35 same range
}

# Friendly sheet names for each variable
SHEET_NAMES = {
    "temp_diff": "Temp Difference (YoY)",
    "min_temp_2023": "Min Temp 2023",
    "max_temp_2023": "Max Temp 2023",
    "avg_temp_2023": "Avg Temp 2023",
    "min_temp_2024": "Min Temp 2024",
    "max_temp_2024": "Max Temp 2024",
    "avg_temp_2024": "Avg Temp 2024",
}


def build_sheet_df(
    week_preds: dict[int, np.ndarray],
    target_df: pd.DataFrame,
    name_col: str,
) -> pd.DataFrame:
    """Build a single-variable DataFrame with readable date-range columns."""
    data = {
        name_col: target_df[name_col].to_numpy(),
        "latitude": target_df["latitude"].to_numpy(),
        "longitude": target_df["longitude"].to_numpy(),
    }
    for week in sorted(week_preds):
        label = WEEK_LABELS.get(week, f"Week {week}")
        data[label] = week_preds[week]

    return pd.DataFrame(data)


def main() -> None:
    cities, districts = load_power_unit_coords()
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    # Precompute target arrays
    city_lons = cities["longitude"].to_numpy()
    city_lats = cities["latitude"].to_numpy()
    dist_lons = districts["longitude"].to_numpy()
    dist_lats = districts["latitude"].to_numpy()

    # Store all predictions: {var_name: {week: array}}
    city_preds: dict[str, dict[int, np.ndarray]] = {}
    dist_preds: dict[str, dict[int, np.ndarray]] = {}

    # --- 1. Temperature difference ---
    print("\n--- Loading temperature difference data ---")
    temp_diff = pd.read_excel(TEMP_DIFF_PATH, engine="calamine")
    temp_diff["week_number"] = temp_diff["week_number"].astype(int)
    print(f"  {temp_diff[['longitude','latitude']].drop_duplicates().shape[0]} stations, "
          f"weeks {sorted(temp_diff['week_number'].unique())}")

    print("\nPredicting temp_diff at cities...")
    city_preds["temp_diff"] = predict_variable(
        temp_diff, "temp_diff", city_lons, city_lats, transformer, "temp_diff")
    print("Predicting temp_diff at districts...")
    dist_preds["temp_diff"] = predict_variable(
        temp_diff, "temp_diff", dist_lons, dist_lats, transformer, "temp_diff")

    # --- 2. Min / Max / Avg temp per year ---
    print("\n--- Loading weekly temperature data (min/max/avg) ---")
    weekly = pd.read_excel(WEEKLY_TEMP_PATH, engine="calamine")
    weekly["week_number"] = weekly["week_number"].astype(int)
    print(f"  {weekly[['longitude','latitude']].drop_duplicates().shape[0]} stations, "
          f"years {sorted(weekly['year'].unique())}, "
          f"weeks {sorted(weekly['week_number'].unique())}")

    for year in YEARS:
        year_data = weekly[weekly["year"] == year].copy()
        if year_data.empty:
            print(f"  No data for {year}, skipping.")
            continue

        for var in WEEKLY_VARS:
            col_name = f"{var}_{year}"
            print(f"\nPredicting {col_name} at cities...")
            city_preds[col_name] = predict_variable(
                year_data, var, city_lons, city_lats, transformer, col_name)
            print(f"Predicting {col_name} at districts...")
            dist_preds[col_name] = predict_variable(
                year_data, var, dist_lons, dist_lats, transformer, col_name)

    # --- Build and write output: one sheet per variable ---
    print(f"\n--- Writing output ({len(city_preds)} sheets per file) ---")

    with pd.ExcelWriter(CITY_OUTPUT) as writer:
        for var_name, week_preds in city_preds.items():
            sheet_name = SHEET_NAMES.get(var_name, var_name)
            df = build_sheet_df(week_preds, cities, "地市供电单位名称")
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  {sheet_name}: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"Saved: {CITY_OUTPUT}")

    with pd.ExcelWriter(DISTRICT_OUTPUT) as writer:
        for var_name, week_preds in dist_preds.items():
            sheet_name = SHEET_NAMES.get(var_name, var_name)
            df = build_sheet_df(week_preds, districts, "区县供电单位名称")
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  {sheet_name}: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"Saved: {DISTRICT_OUTPUT}")


if __name__ == "__main__":
    main()

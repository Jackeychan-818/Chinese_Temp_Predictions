"""
Predict temperature differences using kriging and save map-only plots.

Requirements:
  - pandas
  - numpy
  - pyproj
  - pykrige
  - matplotlib
  - openpyxl

Install (example):
  pip install pandas numpy pyproj pykrige matplotlib openpyxl
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pykrige.ok import OrdinaryKriging
from pyproj import Transformer

DATA_PATH = "/Users/jackey/Desktop/Kriging/Corrected_City-Specific_Weekly_Temperature_Differences_with_Location.xlsx"
GRID_PATH = "/Users/jackey/Desktop/Kriging/china_grid2.xlsx"
OUTPUT_DIR = "/Users/jackey/Desktop/Chinese_Temp_Predictions/Results/Map_Only"

WEEKS = range(22, 35)  # 22–34 inclusive


def require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def main() -> None:
    data = pd.read_excel(DATA_PATH)
    grid = pd.read_excel(GRID_PATH)

    require_columns(data, ["week_number", "longitude", "latitude", "temp_diff"], "data")
    require_columns(grid, ["longitude", "latitude"], "grid")

    # Ensure week_number is string for consistent filtering
    data["week_number"] = data["week_number"].astype(str)

    # Prepare output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    grid_lon = grid["longitude"].to_numpy()
    grid_lat = grid["latitude"].to_numpy()
    grid_x, grid_y = transformer.transform(grid_lon, grid_lat)

    for week in WEEKS:
        week_str = str(week)
        week_data = data[data["week_number"] == week_str]
        if week_data.empty:
            print(f"Week {week} skipped (no data).")
            continue

        lon = week_data["longitude"].to_numpy()
        lat = week_data["latitude"].to_numpy()
        values = week_data["temp_diff"].to_numpy()

        x, y = transformer.transform(lon, lat)

        # Ordinary Kriging with spherical variogram (PyKrige fits parameters internally)
        ok = OrdinaryKriging(
            x,
            y,
            values,
            variogram_model="spherical",
            verbose=False,
            enable_plotting=False,
        )

        z, _ = ok.execute("points", grid_x, grid_y)
        z = np.asarray(z)

        # Plot map-only prediction
        fig, ax = plt.subplots(figsize=(10, 7))
        vmin = np.nanmin(z)
        vmax = np.nanmax(z)
        norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)

        sc = ax.scatter(
            grid_lon,
            grid_lat,
            c=z,
            s=6,
            cmap="bwr",
            norm=norm,
            linewidths=0,
        )
        ax.set_title(f"Predicted Temperature Difference - Week {week}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="box")
        plt.colorbar(sc, ax=ax, label="Predicted temp_diff")

        output_path = os.path.join(OUTPUT_DIR, f"Predicted_TempDiff_Week_{week}.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close(fig)

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

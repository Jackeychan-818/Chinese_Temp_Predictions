"""
Predict weekly temperature differences for China power supply units.

Pipeline:
1. Read power supply units at city level and district/county level.
2. Geocode each unique location with OpenAI API (with local cache reuse).
3. Predict week 22-34 temperature differences by ordinary kriging.
4. Save separate English outputs for city and district/county levels.

Environment:
- Set OPENAI_API_KEY before running.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from openai import OpenAI
from pykrige.ok import OrdinaryKriging
from pyproj import Transformer

PSU_PATH = "/Users/jackey/Desktop/Chinese_Temp_Predictions/power_supply_units.xlsx"
TEMP_DATA_PATH = "/Users/jackey/Desktop/Kriging/Corrected_City-Specific_Weekly_Temperature_Differences_with_Location.xlsx"
OUTPUT_DIR = "/Users/jackey/Desktop/Chinese_Temp_Predictions/Power_Supply_Predictions"
GEOCODE_CACHE_PATH = "/Users/jackey/Desktop/Chinese_Temp_Predictions/geocode_cache_openai.xlsx"

# Recommended default model for this task: good cost/quality trade-off.
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
WEEKS = range(22, 35)
SUPPLY_SUFFIXES = ["供电分公司", "供电服务中心", "供电中心", "供电公司"]


def clean_text(value: object) -> Optional[str]:
    """Normalize text values from Excel."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def normalize_key_value(value: object) -> str:
    """Convert nullable key values to merge-safe strings."""
    if pd.isna(value) or value is None:
        return ""
    return str(value).strip()


def strip_supply_suffix(name: object) -> Optional[str]:
    """Remove power-supply organizational suffixes."""
    text = clean_text(name)
    if not text:
        return None
    for suffix in SUPPLY_SUFFIXES:
        text = text.replace(suffix, "")
    text = text.strip()
    return text or None


def extract_province(province_company_name: object) -> Optional[str]:
    """Extract province name from province company name."""
    text = clean_text(province_company_name)
    if not text:
        return None
    text = text.replace("国网", "").replace("电力公司", "").strip()
    return text or None


def build_city_level_locations(psu: pd.DataFrame) -> pd.DataFrame:
    """Create unique city-level geocoding targets."""
    city = psu[["省公司", "地市供电单位名称"]].drop_duplicates().copy()
    city["level"] = "city"
    city["province_cn"] = city["省公司"].apply(extract_province)
    city["city_cn"] = city["地市供电单位名称"].apply(strip_supply_suffix)
    city["district_cn"] = ""
    city["province_company_cn"] = city["省公司"].apply(clean_text)
    city["city_unit_cn"] = city["地市供电单位名称"].apply(clean_text)
    city["district_unit_cn"] = ""
    return city[
        [
            "level",
            "province_cn",
            "city_cn",
            "district_cn",
            "province_company_cn",
            "city_unit_cn",
            "district_unit_cn",
        ]
    ].copy()


def build_district_level_locations(psu: pd.DataFrame) -> pd.DataFrame:
    """Create unique district/county-level geocoding targets."""
    district = psu[["省公司", "地市供电单位名称", "区县供电单位名称"]].drop_duplicates().copy()
    district["level"] = "district"
    district["province_cn"] = district["省公司"].apply(extract_province)
    district["city_cn"] = district["地市供电单位名称"].apply(strip_supply_suffix)
    district["district_cn"] = district["区县供电单位名称"].apply(strip_supply_suffix)
    district["province_company_cn"] = district["省公司"].apply(clean_text)
    district["city_unit_cn"] = district["地市供电单位名称"].apply(clean_text)
    district["district_unit_cn"] = district["区县供电单位名称"].apply(clean_text)
    return district[
        [
            "level",
            "province_cn",
            "city_cn",
            "district_cn",
            "province_company_cn",
            "city_unit_cn",
            "district_unit_cn",
        ]
    ].copy()


def load_cache(cache_path: Path) -> pd.DataFrame:
    """Load geocode cache if it exists, otherwise return an empty table."""
    if not cache_path.exists():
        return pd.DataFrame(
            columns=[
                "level",
                "province_cn",
                "city_cn",
                "district_cn",
                "province_en",
                "city_en",
                "district_en",
                "latitude",
                "longitude",
                "geocode_status",
                "geocode_confidence",
                "geocode_query",
                "geocode_model",
                "error_message",
                "updated_at",
            ]
        )
    return pd.read_excel(cache_path, engine="calamine")


def save_cache(cache: pd.DataFrame, cache_path: Path) -> None:
    """Persist geocode cache."""
    cache.to_excel(cache_path, index=False)


def make_cache_key(row: pd.Series) -> tuple:
    """Build a stable cache key for location row."""
    return (
        normalize_key_value(row["level"]),
        normalize_key_value(row["province_cn"]),
        normalize_key_value(row["city_cn"]),
        normalize_key_value(row["district_cn"]),
    )


def geocode_with_openai(
    client: OpenAI,
    model: str,
    province_cn: Optional[str],
    city_cn: Optional[str],
    district_cn: Optional[str],
) -> dict:
    """Geocode one location via OpenAI model with JSON-only output."""
    query_parts = [province_cn, city_cn, district_cn, "中国"]
    query_text = " ".join([part for part in query_parts if part])

    if not province_cn or not city_cn:
        return {
            "province_en": None,
            "city_en": None,
            "district_en": None,
            "latitude": None,
            "longitude": None,
            "geocode_status": "invalid_input",
            "geocode_confidence": 0.0,
            "geocode_query": query_text,
            "geocode_model": model,
            "error_message": "Missing province or city.",
        }

    system_prompt = (
        "You are a geocoding assistant for Chinese administrative locations. "
        "Return only valid JSON with no markdown."
    )
    user_prompt = (
        "Find one best latitude/longitude for this place in China.\n"
        f"province_cn: {province_cn}\n"
        f"city_cn: {city_cn}\n"
        f"district_cn: {district_cn}\n\n"
        "Return JSON fields exactly:\n"
        "province_en (string or null), city_en (string or null), district_en (string or null),\n"
        "latitude (number or null), longitude (number or null),\n"
        "status (found|not_found), confidence (0..1), error_message (string or null).\n"
        "If uncertain, set status=not_found and coordinates=null."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
    except Exception as exc:
        return {
            "province_en": None,
            "city_en": None,
            "district_en": None,
            "latitude": None,
            "longitude": None,
            "geocode_status": "error",
            "geocode_confidence": 0.0,
            "geocode_query": query_text,
            "geocode_model": model,
            "error_message": str(exc),
        }

    lat = parsed.get("latitude")
    lon = parsed.get("longitude")
    status = parsed.get("status", "not_found")
    if lat is not None and lon is not None:
        try:
            lat = float(lat)
            lon = float(lon)
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                status = "not_found"
                lat, lon = None, None
        except (TypeError, ValueError):
            status = "not_found"
            lat, lon = None, None
    else:
        lat, lon = None, None

    confidence = parsed.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return {
        "province_en": parsed.get("province_en"),
        "city_en": parsed.get("city_en"),
        "district_en": parsed.get("district_en"),
        "latitude": lat,
        "longitude": lon,
        "geocode_status": status,
        "geocode_confidence": confidence,
        "geocode_query": query_text,
        "geocode_model": model,
        "error_message": parsed.get("error_message"),
    }


def geocode_locations_with_cache(
    locations: pd.DataFrame,
    client: OpenAI,
    model: str,
    cache_path: Path,
    sleep_seconds: float,
) -> pd.DataFrame:
    """Geocode locations using cache + OpenAI API."""
    cache = load_cache(cache_path)
    required_cols = {
        "level",
        "province_cn",
        "city_cn",
        "district_cn",
        "province_en",
        "city_en",
        "district_en",
        "latitude",
        "longitude",
        "geocode_status",
        "geocode_confidence",
        "geocode_query",
        "geocode_model",
        "error_message",
        "updated_at",
    }
    missing_cols = required_cols.difference(cache.columns)
    for col in missing_cols:
        cache[col] = None

    cache_key_to_row = {}
    if not cache.empty:
        for _, cached_row in cache.iterrows():
            key = (
                normalize_key_value(cached_row["level"]),
                normalize_key_value(cached_row["province_cn"]),
                normalize_key_value(cached_row["city_cn"]),
                normalize_key_value(cached_row["district_cn"]),
            )
            cache_key_to_row[key] = cached_row.to_dict()

    results = []
    total = len(locations)

    for index, row in locations.iterrows():
        key = make_cache_key(row)
        if key in cache_key_to_row:
            cached = cache_key_to_row[key].copy()
            cached["level"] = normalize_key_value(cached.get("level"))
            cached["province_cn"] = normalize_key_value(cached.get("province_cn"))
            cached["city_cn"] = normalize_key_value(cached.get("city_cn"))
            cached["district_cn"] = normalize_key_value(cached.get("district_cn"))
            cached["cache_hit"] = True
            results.append(cached)
            continue

        geocoded = geocode_with_openai(
            client=client,
            model=model,
            province_cn=row["province_cn"],
            city_cn=row["city_cn"],
            district_cn=row["district_cn"],
        )
        geocoded_row = {
            "level": row["level"],
            "province_cn": row["province_cn"],
            "city_cn": row["city_cn"],
            "district_cn": row["district_cn"],
            "province_en": geocoded["province_en"],
            "city_en": geocoded["city_en"],
            "district_en": geocoded["district_en"],
            "latitude": geocoded["latitude"],
            "longitude": geocoded["longitude"],
            "geocode_status": geocoded["geocode_status"],
            "geocode_confidence": geocoded["geocode_confidence"],
            "geocode_query": geocoded["geocode_query"],
            "geocode_model": geocoded["geocode_model"],
            "error_message": geocoded["error_message"],
            "updated_at": pd.Timestamp.utcnow().isoformat(),
            "cache_hit": False,
        }
        results.append(geocoded_row)
        cache_key_to_row[key] = geocoded_row

        progress = len(results)
        if progress % 25 == 0 or progress == total:
            print(f"Geocoding progress: {progress}/{total}")

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    merged_cache = pd.DataFrame(cache_key_to_row.values())
    save_cache(merged_cache, cache_path)
    print(f"Saved geocode cache: {cache_path}")

    geocoded_df = pd.DataFrame(results)
    return geocoded_df


def run_kriging_predictions(
    geocoded_locations: pd.DataFrame,
    temp_data: pd.DataFrame,
    weeks: range,
) -> pd.DataFrame:
    """Predict weekly temperature difference and temperatures with ordinary kriging."""
    valid = geocoded_locations[
        geocoded_locations["latitude"].notna() & geocoded_locations["longitude"].notna()
    ].copy()

    if valid.empty:
        return valid

    # Expected temperature columns after normalization in main().
    required_cols = ["temp_diff", "temp_2023", "temp_2024"]
    missing = [col for col in required_cols if col not in temp_data.columns]
    if missing:
        print(f"Missing temperature columns in source data: {missing}")
        for week in weeks:
            valid[f"week_{week}_pred"] = np.nan
            valid[f"week_{week}_temp_2023_pred"] = np.nan
            valid[f"week_{week}_temp_2024_pred"] = np.nan
        return valid

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    target_x, target_y = transformer.transform(
        valid["longitude"].to_numpy(),
        valid["latitude"].to_numpy(),
    )

    for week in weeks:
        diff_col = f"week_{week}_pred"
        t2023_col = f"week_{week}_temp_2023_pred"
        t2024_col = f"week_{week}_temp_2024_pred"

        week_data = temp_data[temp_data["week_number"] == str(week)].copy()
        week_data = week_data.dropna(subset=["longitude", "latitude"])
        if len(week_data) < 3:
            valid[diff_col] = np.nan
            valid[t2023_col] = np.nan
            valid[t2024_col] = np.nan
            print(f"Week {week}: skipped, not enough training points")
            continue

        # Predict each field independently using the same kriging setup.
        field_map = {
            "temp_diff": diff_col,
            "temp_2023": t2023_col,
            "temp_2024": t2024_col,
        }
        week_messages = []

        for source_col, out_col in field_map.items():
            field_data = week_data.dropna(subset=[source_col]).copy()
            if len(field_data) < 3:
                valid[out_col] = np.nan
                week_messages.append(f"{source_col}=insufficient_data")
                continue

            try:
                train_x, train_y = transformer.transform(
                    field_data["longitude"].to_numpy(),
                    field_data["latitude"].to_numpy(),
                )
                values = field_data[source_col].to_numpy(dtype=float)

                ok = OrdinaryKriging(
                    train_x,
                    train_y,
                    values,
                    variogram_model="spherical",
                    verbose=False,
                    enable_plotting=False,
                )
                pred, _ = ok.execute("points", target_x, target_y)
                valid[out_col] = np.asarray(pred)
                week_messages.append(f"{source_col}=ok")
            except Exception as exc:
                valid[out_col] = np.nan
                week_messages.append(f"{source_col}=failed({exc})")

        print(f"Week {week}: " + ", ".join(week_messages))

    return valid


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Predict weekly temperatures at city and district/county levels."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_OPENAI_MODEL,
        help="OpenAI model for geocoding (default: gpt-4.1-mini).",
    )
    parser.add_argument(
        "--limit-city",
        type=int,
        default=None,
        help="Optional test limit for city-level rows.",
    )
    parser.add_argument(
        "--limit-district",
        type=int,
        default=None,
        help="Optional test limit for district-level rows.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.2,
        help="Delay between API calls to avoid burst traffic.",
    )
    return parser.parse_args()


def main() -> None:
    """Run full location geocoding + kriging prediction workflow."""
    args = parse_args()
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Export it before running.")

    print("Loading power supply units...")
    psu = pd.read_excel(PSU_PATH, engine="calamine")

    city_locations = build_city_level_locations(psu)
    district_locations = build_district_level_locations(psu)

    if args.limit_city is not None:
        city_locations = city_locations.head(args.limit_city).copy()
    if args.limit_district is not None:
        district_locations = district_locations.head(args.limit_district).copy()

    print(f"City-level locations: {len(city_locations)}")
    print(f"District/county-level locations: {len(district_locations)}")
    print(f"Geocoding model: {args.model}")

    client = OpenAI(api_key=api_key)
    cache_path = Path(GEOCODE_CACHE_PATH)

    print("\nGeocoding city-level locations...")
    geocoded_city = geocode_locations_with_cache(
        locations=city_locations,
        client=client,
        model=args.model,
        cache_path=cache_path,
        sleep_seconds=args.sleep_seconds,
    )
    city_full = city_locations.merge(
        geocoded_city.drop(columns=["cache_hit"], errors="ignore"),
        on=["level", "province_cn", "city_cn", "district_cn"],
        how="left",
    )

    print("\nGeocoding district/county-level locations...")
    geocoded_district = geocode_locations_with_cache(
        locations=district_locations,
        client=client,
        model=args.model,
        cache_path=cache_path,
        sleep_seconds=args.sleep_seconds,
    )
    district_full = district_locations.merge(
        geocoded_district.drop(columns=["cache_hit"], errors="ignore"),
        on=["level", "province_cn", "city_cn", "district_cn"],
        how="left",
    )

    print("\nLoading temperature data...")
    temp_data = pd.read_excel(TEMP_DATA_PATH, engine="calamine")
    temp_data["week_number"] = temp_data["week_number"].astype(str)
    # Normalize year columns to English-safe names.
    rename_map = {}
    if 2023 in temp_data.columns:
        rename_map[2023] = "temp_2023"
    if "2023" in temp_data.columns:
        rename_map["2023"] = "temp_2023"
    if 2024 in temp_data.columns:
        rename_map[2024] = "temp_2024"
    if "2024" in temp_data.columns:
        rename_map["2024"] = "temp_2024"
    if rename_map:
        temp_data = temp_data.rename(columns=rename_map)

    print("\nPredicting city-level temperatures...")
    city_pred = run_kriging_predictions(city_full, temp_data, WEEKS)

    print("\nPredicting district/county-level temperatures...")
    district_pred = run_kriging_predictions(district_full, temp_data, WEEKS)

    city_output = output_dir / "city_level_temp_predictions.xlsx"
    district_output = output_dir / "district_level_temp_predictions.xlsx"
    city_pred.to_excel(city_output, index=False)
    district_pred.to_excel(district_output, index=False)

    summary = pd.DataFrame(
        [
            {
                "level": "city",
                "total_locations": len(city_full),
                "geocoded_locations": int(city_full["latitude"].notna().sum()),
                "predicted_locations": len(city_pred),
            },
            {
                "level": "district",
                "total_locations": len(district_full),
                "geocoded_locations": int(district_full["latitude"].notna().sum()),
                "predicted_locations": len(district_pred),
            },
        ]
    )
    summary_output = output_dir / "prediction_summary.xlsx"
    summary.to_excel(summary_output, index=False)

    print("\n=== Run Summary ===")
    print(summary.to_string(index=False))
    print(f"\nSaved city predictions: {city_output}")
    print(f"Saved district predictions: {district_output}")
    print(f"Saved summary: {summary_output}")


if __name__ == "__main__":
    main()

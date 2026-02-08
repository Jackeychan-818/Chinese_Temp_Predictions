"""
Geocode parsed city/district names to lat/lng coordinates.

Uses cpca (Chinese Province City Area) built-in database as primary lookup,
with a static CSV fallback from sfyc23/China-zip-code-latitude-and-longitude.
"""

from __future__ import annotations

import csv
import ssl
import urllib.request
from collections import defaultdict
from pathlib import Path

import pandas as pd

import cpca

INPUT_FILE = Path("/Users/jackey/Desktop/Chinese_Temp_Predictions/real_city_districts_parsed.xlsx")
OUTPUT_FILE = Path("/Users/jackey/Desktop/Chinese_Temp_Predictions/geocoded_city_districts.xlsx")
FALLBACK_CSV = Path("/Users/jackey/Desktop/Chinese_Temp_Predictions/region_geo_cache.csv")
FALLBACK_URL = (
    "https://raw.githubusercontent.com/sfyc23/China-zip-code-latitude-and-longitude"
    "/master/data/China-City-List-latest.csv"
)

# Province short-name to full-name mapping for adcode lookups.
# Keys are the cleaned province hints from the parsed data (via normalize_province).
PROVINCE_SHORT_TO_FULL = {
    "北京": "北京市",
    "天津": "天津市",
    "上海": "上海市",
    "重庆": "重庆市",
    "河北": "河北省",
    "冀北": "河北省",
    "山西": "山西省",
    "辽宁": "辽宁省",
    "吉林": "吉林省",
    "黑龙江": "黑龙江省",
    "江苏": "江苏省",
    "浙江": "浙江省",
    "安徽": "安徽省",
    "福建": "福建省",
    "江西": "江西省",
    "山东": "山东省",
    "河南": "河南省",
    "湖北": "湖北省",
    "湖南": "湖南省",
    "广东": "广东省",
    "海南": "海南省",
    "四川": "四川省",
    "贵州": "贵州省",
    "云南": "云南省",
    "陕西": "陕西省",
    "甘肃": "甘肃省",
    "青海": "青海省",
    "台湾": "台湾省",
    "内蒙古": "内蒙古自治区",
    "蒙东": "内蒙古自治区",
    "广西": "广西壮族自治区",
    "西藏": "西藏自治区",
    "宁夏": "宁夏回族自治区",
    "新疆": "新疆维吾尔自治区",
    "香港": "香港特别行政区",
    "澳门": "澳门特别行政区",
}

# Manual coordinate overrides for names that can't be matched via cpca suffix rules.
# These are real cities/areas that use abbreviated or non-standard names.
MANUAL_COORDS: dict[str, tuple[float, float]] = {
    "亦庄": (39.795, 116.506),
    "城南": (39.084, 117.200),       # 天津城南
    "市区": (31.230, 121.474),       # generic 市区, use Shanghai center
    "胜利": (37.462, 118.515),       # 胜利油田/东营
    "两江": (29.563, 106.552),       # 重庆两江新区
    "天府新区": (30.456, 104.070),
    "雄安新区": (38.980, 115.960),
    "西咸": (34.330, 108.710),       # 西咸新区
    "宁东": (38.050, 106.600),       # 宁东能源化工基地
    "黄化": (36.480, 102.000),       # 黄南+化隆 area in Qinghai
    "克州": (39.715, 76.168),        # 克孜勒苏柯尔克孜自治州
    "伊犁伊河": (43.921, 81.330),    # 伊犁
    "宝泉岭": (47.350, 130.570),     # 黑龙江农垦
    "红兴隆": (46.400, 131.500),     # 黑龙江农垦
    "建三江": (47.240, 132.520),     # 黑龙江农垦
    "兴凯湖": (45.350, 132.400),     # 黑龙江农垦
    "九三": (48.820, 125.180),       # 黑龙江农垦
}

# Administrative suffixes to try when matching names.
ADMIN_SUFFIXES = [
    "",  # exact match first
    "市", "区", "县", "州", "盟", "旗", "地区",
    "自治县", "自治旗", "自治州",
    "满族自治县", "回族自治县", "土家族苗族自治县",
    "蒙古族自治县", "彝族自治县", "藏族自治县",
    "苗族自治县", "侗族自治县", "瑶族自治县",
    "哈尼族彝族自治县", "壮族瑶族自治县",
    "苗族侗族自治县", "布依族苗族自治县",
    "土家族苗族自治州", "藏族羌族自治州",
    "哈萨克自治州", "蒙古自治州",
    "藏族自治州", "彝族自治州",
    "苗族侗族自治州", "布依族苗族自治州",
    "壮族苗族自治州",
    "林区", "矿区", "特区",
    "新区", "开发区",
]


def normalize_province(province_raw: str) -> str:
    """Strip company/org wrappers from province column to get a clean province hint."""
    text = str(province_raw).strip() if pd.notna(province_raw) else ""
    if not text:
        return ""
    for token in [
        "国网", "国家电网", "南方电网",
        "内蒙古电力集团", "内蒙古电力",
        "省电力公司", "电力有限公司", "电力有限责任公司",
        "电力公司", "有限责任公司", "有限公司", "公司",
        "壮族自治区", "回族自治区", "维吾尔自治区",
        "自治区", "特别行政区", "省", "市",
    ]:
        text = text.replace(token, "")
    return text.strip()


def build_cpca_lookup() -> tuple[
    dict[str, list[dict]],
    dict[str, str],
]:
    """Build lookup dicts from cpca's built-in database.

    Returns:
        name_to_entries: name -> list of {name, adcode, lat, lng, rank, province}
        adcode_to_province: adcode prefix (2 digits) -> province name
    """
    d = cpca.ad_2_addr_dict

    # First pass: build adcode->province map (rank 0 entries)
    adcode_to_province: dict[str, str] = {}
    for _k, v in d.items():
        if v.rank == 0:
            adcode_to_province[v.adcode[:2]] = v.name

    # Second pass: build name -> entries map
    name_to_entries: dict[str, list[dict]] = defaultdict(list)
    for _k, v in d.items():
        lat = v.latitude
        lng = v.longitude
        if not lat or not lng:
            continue
        province = adcode_to_province.get(v.adcode[:2], "")
        entry = {
            "name": v.name,
            "adcode": v.adcode,
            "latitude": float(lat),
            "longitude": float(lng),
            "rank": v.rank,
            "province": province,
        }
        name_to_entries[v.name].append(entry)

    return dict(name_to_entries), adcode_to_province


def download_fallback_csv(path: Path) -> None:
    """Download the static CSV fallback file if not already cached."""
    if path.exists():
        return
    print(f"Downloading fallback CSV to {path} ...")
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(FALLBACK_URL)
        with urllib.request.urlopen(req, context=ctx) as resp:
            path.write_bytes(resp.read())
        print("  Done.")
    except Exception as e:
        print(f"  Warning: failed to download fallback CSV: {e}")
        print("  Continuing with cpca-only matching.")


def build_fallback_lookup(path: Path) -> dict[str, list[dict]]:
    """Build a name->entries lookup from the fallback CSV."""
    if not path.exists():
        return {}

    name_to_entries: dict[str, list[dict]] = defaultdict(list)

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # CSV columns vary; try common column names
            name = row.get("name", "") or row.get("地名", "") or row.get("Name", "")
            lat_str = row.get("lat", "") or row.get("latitude", "") or row.get("纬度", "")
            lng_str = row.get("lng", "") or row.get("longitude", "") or row.get("经度", "")
            province = row.get("province", "") or row.get("省份", "") or ""

            if not name or not lat_str or not lng_str:
                continue
            try:
                lat = float(lat_str)
                lng = float(lng_str)
            except ValueError:
                continue

            entry = {
                "name": name,
                "latitude": lat,
                "longitude": lng,
                "province": province,
                "rank": -1,
                "adcode": "",
            }
            name_to_entries[name].append(entry)

    return dict(name_to_entries)


def province_matches(entry_province: str, province_hint: str) -> bool:
    """Check if an entry's province matches the province hint."""
    if not province_hint or not entry_province:
        return True  # no constraint
    full = PROVINCE_SHORT_TO_FULL.get(province_hint, province_hint)
    # Direct substring match either way
    return (province_hint in entry_province
            or full in entry_province
            or entry_province in full
            or entry_province in province_hint)


def pick_best_entry(
    entries: list[dict],
    province_hint: str,
    prefer_rank: int | None = None,
) -> dict | None:
    """Pick the best entry from a list, using province hint for disambiguation."""
    if not entries:
        return None
    if len(entries) == 1:
        return entries[0]

    # Filter by province
    matched = [e for e in entries if province_matches(e["province"], province_hint)]
    if not matched:
        matched = entries  # fall back to all

    if len(matched) == 1:
        return matched[0]

    # Prefer specific rank if given
    if prefer_rank is not None:
        rank_matched = [e for e in matched if e["rank"] == prefer_rank]
        if rank_matched:
            matched = rank_matched

    # Among remaining, prefer rank 1 (city-level) for cities, rank 2 (district) for districts
    # Default: pick first
    return matched[0]


def geocode_name(
    name: str,
    province_hint: str,
    cpca_lookup: dict[str, list[dict]],
    fallback_lookup: dict[str, list[dict]],
    prefer_rank: int | None = None,
) -> tuple[float, float, str] | None:
    """Try to geocode a single name.

    Returns (latitude, longitude, match_source) or None.
    """
    if not name:
        return None

    # Already has an admin suffix? Try exact first.
    has_suffix = any(name.endswith(s) for s in ADMIN_SUFFIXES if s)

    # Strategy 1: cpca lookup with suffix appending
    suffixes_to_try = [""] if has_suffix else ADMIN_SUFFIXES
    if has_suffix:
        # Also try with other suffixes after exact match
        suffixes_to_try = [""] + [s for s in ADMIN_SUFFIXES if s]

    for suffix in suffixes_to_try:
        candidate = name + suffix if suffix else name
        entries = cpca_lookup.get(candidate)
        if entries:
            best = pick_best_entry(entries, province_hint, prefer_rank)
            if best:
                source = "cpca_exact" if not suffix else f"cpca_suffix_{suffix}"
                return best["latitude"], best["longitude"], source

    # Strategy 2: for names that already end with a suffix, try stripping it
    # and re-trying with different suffixes (e.g., "恩施" parsed from "恩施州")
    if has_suffix:
        for s in ADMIN_SUFFIXES:
            if s and name.endswith(s):
                base = name[: -len(s)]
                if len(base) >= 2:
                    for suffix in ADMIN_SUFFIXES:
                        candidate = base + suffix if suffix else base
                        if candidate == name:
                            continue
                        entries = cpca_lookup.get(candidate)
                        if entries:
                            best = pick_best_entry(entries, province_hint, prefer_rank)
                            if best:
                                return best["latitude"], best["longitude"], f"cpca_resuffix_{s}_to_{suffix}"
                break

    # Strategy 3: substring match in cpca — find entries whose name starts with our name
    # This catches ethnic-minority autonomous prefectures/counties like
    # "石柱" -> "石柱土家族自治县", "延边" -> "延边朝鲜族自治州"
    if len(name) >= 2:
        candidates = [
            (k, vs) for k, vs in cpca_lookup.items()
            if k.startswith(name) and k != name
        ]
        if candidates:
            # Prefer shorter names (closest match)
            candidates.sort(key=lambda x: len(x[0]))
            for _cand_name, vs in candidates:
                best = pick_best_entry(vs, province_hint, prefer_rank)
                if best:
                    return best["latitude"], best["longitude"], f"cpca_prefix_{_cand_name}"

    # Strategy 4: fallback CSV
    for suffix in suffixes_to_try:
        candidate = name + suffix if suffix else name
        entries = fallback_lookup.get(candidate)
        if entries:
            best = pick_best_entry(entries, province_hint, prefer_rank)
            if best:
                source = "fallback_exact" if not suffix else f"fallback_suffix_{suffix}"
                return best["latitude"], best["longitude"], source

    # Strategy 5: manual overrides
    if name in MANUAL_COORDS:
        lat, lng = MANUAL_COORDS[name]
        return lat, lng, "manual"

    return None


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    # Build lookups
    print("Building cpca lookup ...")
    cpca_lookup, _ = build_cpca_lookup()
    print(f"  {len(cpca_lookup)} unique names in cpca database")

    download_fallback_csv(FALLBACK_CSV)
    print("Building fallback lookup ...")
    fallback_lookup = build_fallback_lookup(FALLBACK_CSV)
    print(f"  {len(fallback_lookup)} unique names in fallback CSV")

    # Read parsed data
    city_df = pd.read_excel(INPUT_FILE, sheet_name="Parsed_City_Units")
    district_df = pd.read_excel(INPUT_FILE, sheet_name="Parsed_District_Units")

    # --- Geocode cities ---
    print("\nGeocoding cities ...")
    city_ok = city_df[city_df["city_parse_status"] == "ok"].copy()
    city_unique = (
        city_ok[["省公司", "parsed_city_name"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    city_rows = []
    for _, row in city_unique.iterrows():
        name = row["parsed_city_name"]
        province_hint = normalize_province(row["省公司"])
        result = geocode_name(name, province_hint, cpca_lookup, fallback_lookup, prefer_rank=1)
        if result:
            lat, lng, source = result
            city_rows.append({
                "parsed_city_name": name,
                "province": province_hint,
                "latitude": lat,
                "longitude": lng,
                "match_source": source,
            })
        else:
            city_rows.append({
                "parsed_city_name": name,
                "province": province_hint,
                "latitude": None,
                "longitude": None,
                "match_source": "unmatched",
            })

    city_coords_df = pd.DataFrame(city_rows)
    city_matched = city_coords_df[city_coords_df["latitude"].notna()]
    city_unmatched = city_coords_df[city_coords_df["latitude"].isna()]
    print(f"  Cities matched: {len(city_matched)} / {len(city_coords_df)}")

    # --- Geocode districts ---
    print("Geocoding districts ...")
    dist_ok = district_df[district_df["district_parse_status"] == "ok"].copy()
    dist_unique = (
        dist_ok[["省公司", "parsed_city_name", "parsed_district_name"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    dist_rows = []
    for _, row in dist_unique.iterrows():
        name = row["parsed_district_name"]
        city_name = row["parsed_city_name"] if pd.notna(row["parsed_city_name"]) else ""
        province_hint = normalize_province(row["省公司"])
        result = geocode_name(name, province_hint, cpca_lookup, fallback_lookup, prefer_rank=2)
        if result:
            lat, lng, source = result
            dist_rows.append({
                "parsed_district_name": name,
                "parsed_city_name": city_name,
                "province": province_hint,
                "latitude": lat,
                "longitude": lng,
                "match_source": source,
            })
        else:
            dist_rows.append({
                "parsed_district_name": name,
                "parsed_city_name": city_name,
                "province": province_hint,
                "latitude": None,
                "longitude": None,
                "match_source": "unmatched",
            })

    dist_coords_df = pd.DataFrame(dist_rows)
    dist_matched = dist_coords_df[dist_coords_df["latitude"].notna()]
    dist_unmatched = dist_coords_df[dist_coords_df["latitude"].isna()]
    print(f"  Districts matched: {len(dist_matched)} / {len(dist_coords_df)}")

    # --- Combine unmatched ---
    unmatched_rows = []
    for _, row in city_unmatched.iterrows():
        unmatched_rows.append({
            "level": "city",
            "parsed_name": row["parsed_city_name"],
            "province": row["province"],
        })
    for _, row in dist_unmatched.iterrows():
        unmatched_rows.append({
            "level": "district",
            "parsed_name": row["parsed_district_name"],
            "parsed_city_name": row.get("parsed_city_name", ""),
            "province": row["province"],
        })
    unmatched_df = pd.DataFrame(unmatched_rows) if unmatched_rows else pd.DataFrame(
        columns=["level", "parsed_name", "parsed_city_name", "province"]
    )

    # --- Write output ---
    with pd.ExcelWriter(OUTPUT_FILE) as writer:
        city_coords_df[city_coords_df["latitude"].notna()].to_excel(
            writer, sheet_name="City_Coordinates", index=False
        )
        dist_coords_df[dist_coords_df["latitude"].notna()].to_excel(
            writer, sheet_name="District_Coordinates", index=False
        )
        unmatched_df.to_excel(writer, sheet_name="Unmatched", index=False)

    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"  City_Coordinates: {len(city_matched)} rows")
    print(f"  District_Coordinates: {len(dist_matched)} rows")
    print(f"  Unmatched: {len(unmatched_df)} rows")

    # Spot-check a few known cities
    print("\n--- Spot checks ---")
    spot_checks = ["成都", "武汉", "长沙", "杭州", "南京", "济南", "西安", "石柱", "延边"]
    for name in spot_checks:
        matches = city_coords_df[city_coords_df["parsed_city_name"] == name]
        if len(matches) > 0:
            r = matches.iloc[0]
            print(f"  {name}: lat={r['latitude']}, lng={r['longitude']} ({r['match_source']})")
        else:
            # Check districts
            d_matches = dist_coords_df[dist_coords_df["parsed_district_name"] == name]
            if len(d_matches) > 0:
                r = d_matches.iloc[0]
                print(f"  {name} (district): lat={r['latitude']}, lng={r['longitude']} ({r['match_source']})")
            else:
                print(f"  {name}: not found in parsed data")


if __name__ == "__main__":
    main()

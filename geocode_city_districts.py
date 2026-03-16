"""
Geocode parsed city/district names to lat/lng coordinates.

Uses cpca (Chinese Province City Area) built-in database as primary lookup,
with a static CSV fallback from sfyc23/China-zip-code-latitude-and-longitude.
"""

from __future__ import annotations

import csv
import re
import ssl
import urllib.request
from collections import defaultdict
from pathlib import Path

import pandas as pd

import cpca

INPUT_FILE = Path("/Users/jackey/Desktop/Chinese_Temp_Predictions/unique_city_districts.xlsx")
PARSED_FILE = Path("/Users/jackey/Desktop/Chinese_Temp_Predictions/real_city_districts_parsed.xlsx")
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

# Org suffixes to strip from raw unit names for direct geocoding attempts.
ORG_SUFFIXES = [
    "客户服务分中心",
    "供电服务中心",
    "供配电中心",
    "供电分公司",
    "供电分局",
    "供电中心",
    "供电公司",
    "供电支公司",
    "供电所",
    "服务所",
    "电力局",
    "电力公司",
    "电力有限公司",
    "电力有限责任公司",
    "有限责任公司",
    "有限公司",
    "分公司",
    "中心",
]

# Patterns that indicate a non-geocodable junk entry.
JUNK_PATTERNS = [
    "供电区域",
    "供电分中心",
    "供电部",
    "大客户",
    "自备电厂",
    "公司本部",
    "局直",
    "本部",
    "（删",
    "(删",
    "待删除",
    "删除",
    "暂停",
]

# Regex to extract city+district from names like "上饶市信州区供电分公司",
# "国网冀北唐山市丰南区供电公司", "南昌市红谷滩供电分公司" etc.
_CITY_DISTRICT_RE = re.compile(
    r"(?:国网|南方电网)?(?:冀北|蒙东)?"
    r"(.{2,6}?[市州])"          # city part
    r"(.{2,6}?[区县市旗])"      # district part
    r"(?:供电|配售电|电力)"
)

# Regex for simpler names like "东营区供电公司" where the name IS the district
_DIRECT_DISTRICT_RE = re.compile(
    r"^(.{2,8}?[区县市旗州盟])"
    r"(?:供电|配售电|电力|客户)"
)


def is_junk_entry(name: str) -> bool:
    """Check if a name is a non-geocodable junk/admin entry."""
    return any(p in name for p in JUNK_PATTERNS)


def extract_district_from_raw(raw_name: str) -> tuple[str, str] | None:
    """Try to extract a district name directly from a raw unit name.

    Returns (district_name_with_suffix, city_name) or None.
    """
    # Try city+district pattern first
    m = _CITY_DISTRICT_RE.search(raw_name)
    if m:
        return m.group(2), m.group(1)

    # Try direct district pattern (e.g. "东营区供电公司")
    m = _DIRECT_DISTRICT_RE.match(raw_name)
    if m:
        return m.group(1), ""

    return None


def strip_org_suffix(raw_name: str) -> str:
    """Strip organizational suffixes to get the place name core."""
    # Remove common prefixes
    text = raw_name
    for prefix in ["国网", "南方电网"]:
        if text.startswith(prefix):
            text = text[len(prefix):]

    # Remove org suffixes (longest first)
    for suffix in ORG_SUFFIXES:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break

    return text.strip()


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
    if not PARSED_FILE.exists():
        raise FileNotFoundError(f"Parsed file not found: {PARSED_FILE}")

    # Build lookups
    print("Building cpca lookup ...")
    cpca_lookup, _ = build_cpca_lookup()
    print(f"  {len(cpca_lookup)} unique names in cpca database")

    download_fallback_csv(FALLBACK_CSV)
    print("Building fallback lookup ...")
    fallback_lookup = build_fallback_lookup(FALLBACK_CSV)
    print(f"  {len(fallback_lookup)} unique names in fallback CSV")

    # Read input data (original unit names)
    unique_cities = pd.read_excel(INPUT_FILE, sheet_name="Unique_Cities")
    unique_districts = pd.read_excel(INPUT_FILE, sheet_name="Unique_Districts")
    city_district_pairs = pd.read_excel(INPUT_FILE, sheet_name="City_District_Pairs")

    # Read parsed data to build raw_name -> (parsed_name, province) mapping
    parsed_city_df = pd.read_excel(PARSED_FILE, sheet_name="Parsed_City_Units")
    parsed_dist_df = pd.read_excel(PARSED_FILE, sheet_name="Parsed_District_Units")

    # Build city mapping: raw_city_unit -> (parsed_city_name, 省公司)
    city_parsed_map: dict[str, tuple[str, str]] = {}
    for _, row in parsed_city_df.iterrows():
        raw = row["raw_city_unit"]
        if raw not in city_parsed_map and row["city_parse_status"] == "ok":
            city_parsed_map[raw] = (row["parsed_city_name"], row["省公司"])

    # Build district mapping: (raw_city_unit, raw_district_unit) -> (parsed_district_name, parsed_city_name, 省公司)
    dist_parsed_map: dict[tuple[str, str], tuple[str, str, str]] = {}
    for _, row in parsed_dist_df.iterrows():
        key = (row["raw_city_unit"], row["raw_district_unit"])
        if key not in dist_parsed_map and row["district_parse_status"] == "ok":
            dist_parsed_map[key] = (
                row["parsed_district_name"],
                row["parsed_city_name"] if pd.notna(row["parsed_city_name"]) else "",
                row["省公司"],
            )

    # --- Geocode Unique_Cities ---
    print("\nGeocoding cities ...")
    city_rows = []
    for _, row in unique_cities.iterrows():
        raw_name = row["地市供电单位名称"]

        # Check junk
        if is_junk_entry(raw_name):
            city_rows.append({
                "地市供电单位名称": raw_name,
                "latitude": None,
                "longitude": None,
                "match_source": "junk",
            })
            continue

        # Strategy A: use parsed mapping
        mapping = city_parsed_map.get(raw_name)
        if mapping:
            parsed_name, province_raw = mapping
            province_hint = normalize_province(province_raw)
            result = geocode_name(parsed_name, province_hint, cpca_lookup, fallback_lookup, prefer_rank=1)
            if result:
                lat, lng, source = result
                city_rows.append({
                    "地市供电单位名称": raw_name,
                    "latitude": lat,
                    "longitude": lng,
                    "match_source": source,
                })
                continue

        # Strategy B: try stripping org suffix from raw name and geocoding directly
        stripped = strip_org_suffix(raw_name)
        if stripped and len(stripped) >= 2:
            province_hint = ""
            if mapping:
                province_hint = normalize_province(mapping[1])
            result = geocode_name(stripped, province_hint, cpca_lookup, fallback_lookup, prefer_rank=1)
            if result:
                lat, lng, source = result
                city_rows.append({
                    "地市供电单位名称": raw_name,
                    "latitude": lat,
                    "longitude": lng,
                    "match_source": f"raw_strip:{source}",
                })
                continue

        city_rows.append({
            "地市供电单位名称": raw_name,
            "latitude": None,
            "longitude": None,
            "match_source": "no_parse" if not mapping else "unmatched",
        })

    city_out_df = pd.DataFrame(city_rows)
    city_matched_count = city_out_df["latitude"].notna().sum()
    print(f"  Cities matched: {city_matched_count} / {len(city_out_df)}")

    # Build city coord lookup for fallback use later
    city_coord_map: dict[str, tuple[float, float, str]] = {}
    for _, r in city_out_df[city_out_df["latitude"].notna()].iterrows():
        city_coord_map[r["地市供电单位名称"]] = (r["latitude"], r["longitude"], r["match_source"])

    # --- Geocode Unique_Districts ---
    print("Geocoding districts ...")
    # Build district -> list of cities from the pairs sheet.
    dist_to_cities: dict[str, list[str]] = defaultdict(list)
    for _, row in city_district_pairs.iterrows():
        dist_to_cities[row["区县供电单位名称"]].append(row["地市供电单位名称"])

    dist_rows = []
    for _, row in unique_districts.iterrows():
        raw_dist = row["区县供电单位名称"]

        # Check junk
        if is_junk_entry(raw_dist):
            dist_rows.append({
                "区县供电单位名称": raw_dist,
                "latitude": None,
                "longitude": None,
                "match_source": "junk",
            })
            continue

        # Gather province hint from any associated city
        province_hint = ""
        for raw_city in dist_to_cities.get(raw_dist, []):
            key = (raw_city, raw_dist)
            mapping = dist_parsed_map.get(key)
            if mapping:
                province_hint = normalize_province(mapping[2])
                break

        # Strategy A: use parsed mapping (original approach)
        result = None
        for raw_city in dist_to_cities.get(raw_dist, []):
            key = (raw_city, raw_dist)
            mapping = dist_parsed_map.get(key)
            if mapping:
                parsed_dist_name = mapping[0]
                prov = normalize_province(mapping[2])
                result = geocode_name(parsed_dist_name, prov, cpca_lookup, fallback_lookup, prefer_rank=2)
                if result:
                    break

        if result:
            lat, lng, source = result
            dist_rows.append({
                "区县供电单位名称": raw_dist,
                "latitude": lat,
                "longitude": lng,
                "match_source": source,
            })
            continue

        # Strategy B: extract city+district directly from raw name
        # e.g. "上饶市信州区供电分公司" -> district="信州区"
        extracted = extract_district_from_raw(raw_dist)
        if extracted:
            dist_name, city_name = extracted
            result = geocode_name(dist_name, province_hint, cpca_lookup, fallback_lookup, prefer_rank=2)
            if result:
                lat, lng, source = result
                dist_rows.append({
                    "区县供电单位名称": raw_dist,
                    "latitude": lat,
                    "longitude": lng,
                    "match_source": f"extract_raw:{source}",
                })
                continue

        # Strategy C: strip org suffix from raw name and try geocoding
        stripped = strip_org_suffix(raw_dist)
        if stripped and len(stripped) >= 2:
            result = geocode_name(stripped, province_hint, cpca_lookup, fallback_lookup, prefer_rank=2)
            if result:
                lat, lng, source = result
                dist_rows.append({
                    "区县供电单位名称": raw_dist,
                    "latitude": lat,
                    "longitude": lng,
                    "match_source": f"raw_strip:{source}",
                })
                continue

        # Strategy D: fall back to parent city coordinates
        city_fallback = None
        for raw_city in dist_to_cities.get(raw_dist, []):
            if raw_city in city_coord_map:
                city_fallback = city_coord_map[raw_city]
                break

        if city_fallback:
            lat, lng, csource = city_fallback
            dist_rows.append({
                "区县供电单位名称": raw_dist,
                "latitude": lat,
                "longitude": lng,
                "match_source": f"city_fallback:{csource}",
            })
            continue

        dist_rows.append({
            "区县供电单位名称": raw_dist,
            "latitude": None,
            "longitude": None,
            "match_source": "unmatched",
        })

    dist_out_df = pd.DataFrame(dist_rows)
    dist_matched_count = dist_out_df["latitude"].notna().sum()
    dist_junk_count = (dist_out_df["match_source"] == "junk").sum()
    dist_city_fb_count = dist_out_df["match_source"].str.startswith("city_fallback").sum()
    dist_extract_count = dist_out_df["match_source"].str.startswith("extract_raw").sum()
    dist_strip_count = dist_out_df["match_source"].str.startswith("raw_strip").sum()
    print(f"  Districts matched: {dist_matched_count} / {len(dist_out_df)}")
    print(f"    - via extract from raw name: {dist_extract_count}")
    print(f"    - via raw name strip: {dist_strip_count}")
    print(f"    - via city fallback: {dist_city_fb_count}")
    print(f"    - junk (skipped): {dist_junk_count}")

    # --- Geocode City_District_Pairs ---
    print("\nGeocoding city-district pairs ...")

    dist_coord_map: dict[str, tuple[float, float, str]] = {}
    for _, r in dist_out_df[dist_out_df["latitude"].notna()].iterrows():
        dist_coord_map[r["区县供电单位名称"]] = (r["latitude"], r["longitude"], r["match_source"])

    pair_rows = []
    for _, row in city_district_pairs.iterrows():
        raw_city = row["地市供电单位名称"]
        raw_dist = row["区县供电单位名称"]

        city_coords = city_coord_map.get(raw_city)
        dist_coords = dist_coord_map.get(raw_dist)

        pair_rows.append({
            "地市供电单位名称": raw_city,
            "city_latitude": city_coords[0] if city_coords else None,
            "city_longitude": city_coords[1] if city_coords else None,
            "city_match_source": city_coords[2] if city_coords else "unmatched",
            "区县供电单位名称": raw_dist,
            "district_latitude": dist_coords[0] if dist_coords else None,
            "district_longitude": dist_coords[1] if dist_coords else None,
            "district_match_source": dist_coords[2] if dist_coords else "unmatched",
        })

    pairs_out_df = pd.DataFrame(pair_rows)
    print(f"  Pairs: {len(pairs_out_df)} rows")

    # --- Write output ---
    with pd.ExcelWriter(OUTPUT_FILE) as writer:
        city_out_df.to_excel(writer, sheet_name="Unique_Cities", index=False)
        dist_out_df.to_excel(writer, sheet_name="Unique_Districts", index=False)
        pairs_out_df.to_excel(writer, sheet_name="City_District_Pairs", index=False)

    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"  Unique_Cities: {len(city_out_df)} rows ({city_matched_count} matched)")
    print(f"  Unique_Districts: {len(dist_out_df)} rows ({dist_matched_count} matched)")
    print(f"  City_District_Pairs: {len(pairs_out_df)} rows")


if __name__ == "__main__":
    main()

"""Build the clean, leakage-free modeling dataset and feature schema for LeadSight."""

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat

REPO_DIR = Path(r"C:\Users\shopu\LeadSight")
DEFAULT_SAV_DIR = Path(
    r"C:\Users\shopu\Downloads\BGD_2025_MICS7_v01_M\BGD_2025_MICS7_v01_M"
    r"\BGD_2025_MICS7_Datasets\Bangladesh MICS7 Datasets"
    r"\Bangladesh MICS7 Datasets\Bangladesh MICS7 SPSS Datasets"
)

def group_floor(val):
    if pd.isna(val):
        return np.nan
    v = int(val)
    if v in {11, 12}:
        return "Natural_Earth_Dung"
    if v == 34:
        return "Finished_Cement"
    if v in {31, 32, 33, 35}:
        return "Finished_Tiles_Other"
    return "Rudimentary_Other"

def group_wall(val):
    if pd.isna(val):
        return np.nan
    v = int(val)
    if v in {11, 12, 13, 14, 21, 22, 23, 35}:
        return "Natural_Mud_Bamboo"
    if v == 27:
        return "Tin"
    if v == 33:
        return "Bricks"
    if v in {31, 32, 34}:
        return "Cement_Concrete"
    return "Other_Wood"

def group_water(val):
    if pd.isna(val):
        return np.nan
    v = int(val)
    if v in {11, 12}:
        return "Piped_Dwelling_Yard"
    if v in {13, 14}:
        return "Piped_Shared_Public"
    if v == 21:
        return "Tubewell_Deep"
    if v == 22:
        return "Tubewell_Shallow"
    return "Other_Water_Source"

def clean_lt_ternary(series):
    """Map MICS LT17/LT18 codes: 1->Yes, 2->No, 8/9->Not_Sure."""
    s = pd.to_numeric(series, errors="coerce")
    return np.select(
        [s == 1.0, s == 2.0, s.isin([8.0, 9.0])],
        ["Yes", "No", "Not_Sure"],
        default="Not_Sure",
    )

def main():
    sav_dir = Path(os.environ.get("MICS_SAV_DIR", str(DEFAULT_SAV_DIR)))
    ch_path = REPO_DIR / "data" / "processed" / "LeadLens_CH_Validated.csv"
    hh_path = sav_dir / "hh.sav"

    ch_cols = [
        "HH1", "HH2", "LN", "PSU", "stratum", "chblweight",
        "C_Lead", "Elevated_Lead", "Elevated_Lead_3_5",
        "CAGE", "HL4", "HH6", "HH7", "HH7A", "melevel",
        "HC4", "HC6A",
        "LT13", "LT15", "LT19B", "LT20",
        "LT17A", "LT17B", "LT17C", "LT17D", "LT17E", "LT17F",
        "LT18A", "LT18B", "LT18C", "LT18D", "LT18E", "LT18F",
        "S_Lead", "S_As", "S_Cd", "S_Hg",
    ]

    ch = pd.read_csv(ch_path, usecols=ch_cols, low_memory=False)
    hh, _ = pyreadstat.read_sav(str(hh_path), usecols=["HH1", "HH2", "HC3", "HC6", "WS1"])

    for frame in (ch, hh):
        for k in ("HH1", "HH2"):
            frame[k] = pd.to_numeric(frame[k], errors="coerce").astype("Int64")

    df = ch.merge(hh, on=["HH1", "HH2"], how="left", validate="many_to_one")
    assert len(df) == 10622, f"Expected 10,622 rows after merge, got {len(df)}"
    assert df["CAGE"].between(12, 59).all(), "CAGE out of validated 12-59m range"

    out = pd.DataFrame(index=df.index)

    # Identifiers, survey design, and targets (NEVER model features)
    meta_cols = ["HH1", "HH2", "LN", "PSU", "stratum", "chblweight"]
    target_cols = ["C_Lead", "Elevated_Lead", "Elevated_Lead_3_5"]
    for c in meta_cols + target_cols:
        out[c] = df[c]

    # 1. Core child, maternal & geographic inputs
    out["child_age_months"] = pd.to_numeric(df["CAGE"], errors="coerce")
    out["child_sex"] = df["HL4"].map({1.0: "Male", 2.0: "Female"})
    out["area_type"] = df["HH6"].map({1.0: "Urban", 2.0: "Rural"})
    out["division_code"] = pd.to_numeric(df["HH7"], errors="coerce").astype("Int64").astype(str)
    out["district_code"] = pd.to_numeric(df["HH7A"], errors="coerce").astype("Int64").astype(str)

    me = pd.to_numeric(df["melevel"], errors="coerce")
    out["mother_education_ord"] = me.where(me < 9.0)

    # 2. Housing & drinking water
    hc3 = pd.to_numeric(df["HC3"], errors="coerce")
    out["sleeping_rooms_capped"] = hc3.where(hc3 < 90.0).clip(upper=4)
    out["floor_group"] = df["HC4"].apply(group_floor)
    out["wall_group"] = df["HC6"].apply(group_wall)
    out["interior_paint"] = df["HC6A"].map({
        1.0: "Good_Paint",
        2.0: "Peeling_Paint",
        3.0: "No_Paint",
    })
    out["water_source_group"] = df["WS1"].apply(group_water)

    # 3. Child behavior & household practices
    out["outdoor_hours"] = pd.to_numeric(df["LT19B"], errors="coerce").clip(lower=0, upper=24)
    out["zero_outdoor_flag"] = (out["outdoor_hours"] == 0).astype(int)

    for raw_c, clean_c in [("LT20", "pica_freq_ord"), ("LT15", "contact_surma_ord"), ("LT13", "amulet_mouth_ord")]:
        s = pd.to_numeric(df[raw_c], errors="coerce")
        out[clean_c] = s.where(s.isin([1.0, 2.0, 3.0, 4.0]))

    # 4. Nearby workshops (LT17A-F) & contact work (LT18A-F)
    lt17_clean = []
    for suffix, name in [
        ("A", "nearby_make_battery"),
        ("B", "nearby_melt_battery"),
        ("C", "nearby_break_battery"),
        ("D", "nearby_jewelry"),
        ("E", "nearby_ewaste"),
        ("F", "nearby_auto_repair"),
    ]:
        out[name] = clean_lt_ternary(df[f"LT17{suffix}"])
        lt17_clean.append(name)

    out["nearby_lead_count"] = (out[lt17_clean] == "Yes").sum(axis=1)
    out["nearby_battery_any"] = (
        out[["nearby_make_battery", "nearby_melt_battery", "nearby_break_battery"]] == "Yes"
    ).any(axis=1).astype(int)

    lt18_clean = []
    for suffix, name in [
        ("A", "contact_make_battery"),
        ("B", "contact_melt_battery"),
        ("C", "contact_break_battery"),
        ("D", "contact_jewelry"),
        ("E", "contact_ewaste"),
        ("F", "contact_auto_repair"),
    ]:
        out[name] = clean_lt_ternary(df[f"LT18{suffix}"])
        lt18_clean.append(name)

    out["contact_lead_count"] = (out[lt18_clean] == "Yes").sum(axis=1)
    out["contact_battery_any"] = (
        out[["contact_make_battery", "contact_melt_battery", "contact_break_battery"]] == "Yes"
    ).any(axis=1).astype(int)

    # 5. Environmental soil comparison features (NEVER used in Parent Model)
    for raw_s in ("S_Lead", "S_As", "S_Cd", "S_Hg"):
        s = pd.to_numeric(df[raw_s], errors="coerce")
        s = s.where((s != 0.0) & (s != 999.0))
        out[raw_s] = s
    out["log_S_Lead"] = np.log1p(out["S_Lead"])
    out["log_S_As"] = np.log1p(out["S_As"])
    out["log_S_Hg"] = np.log1p(out["S_Hg"])
    out["S_Cd_above_LOD"] = np.where(out["S_Cd"].isna(), np.nan, (out["S_Cd"] > 3.54).astype(float))

    parent_numeric = [
        "child_age_months", "mother_education_ord", "sleeping_rooms_capped",
        "outdoor_hours", "zero_outdoor_flag",
        "pica_freq_ord", "contact_surma_ord", "amulet_mouth_ord",
        "nearby_lead_count", "nearby_battery_any",
        "contact_lead_count", "contact_battery_any",
    ]
    parent_categorical = [
        "child_sex", "area_type", "division_code", "district_code",
        "floor_group", "wall_group", "interior_paint", "water_source_group",
    ] + lt17_clean + lt18_clean

    soil_features = ["log_S_Lead", "log_S_As", "log_S_Hg", "S_Cd_above_LOD"]

    schema = {
        "cohort_rows": int(len(out)),
        "age_range_months": [int(out["child_age_months"].min()), int(out["child_age_months"].max())],
        "meta_columns": meta_cols,
        "target_columns": target_cols,
        "primary_target": "Elevated_Lead",
        "secondary_target": "Elevated_Lead_3_5",
        "group_cv_column": "PSU",
        "survey_weight_column": "chblweight",
        "parent_features": {
            "numeric": parent_numeric,
            "categorical": parent_categorical,
            "all": parent_numeric + parent_categorical,
        },
        "soil_env_features": soil_features,
        "excluded_confounders_documented": ["EU1", "EU4", "WS9", "HC7G", "windex5", "HC5", "HC8AF", "LT12", "LT14", "LT16", "LT19A"],
    }

    out_csv = REPO_DIR / "data" / "processed" / "LeadSight_Modeling_Dataset.csv"
    out_schema = REPO_DIR / "docs" / "feature_schema.json"
    out.to_csv(out_csv, index=False)
    out_schema.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    print(f"Saved modeling dataset: {out_csv} | shape={out.shape}")
    print(f"Saved feature schema:   {out_schema}")
    print(f"Parent features: {len(parent_numeric)} numeric + {len(parent_categorical)} categorical = {len(schema['parent_features']['all'])} total")
    print(f"Soil comparison features: {len(soil_features)} (valid soil rows: {out['S_Lead'].notna().sum()})")
    print("\nMissing values in Parent features (non-zero only):")
    miss = out[schema["parent_features"]["all"]].isna().sum()
    print(miss[miss > 0].to_string() if (miss > 0).any() else "None (0 missing)")

if __name__ == "__main__":
    main()

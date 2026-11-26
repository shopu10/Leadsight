"""Milestone 2 & 3: Audit missingness in the validated CH cohort, verify
household key linkage (HH1, HH2), and screen candidate features across
domains from HH.sav without leaking microdata.
"""

import os
from pathlib import Path
import pandas as pd
import pyreadstat

SAV_DIR = Path(os.environ.get("MICS_SAV_DIR", "data/raw"))
PROCESSED_DIR = Path("data/processed")
DOCS_DIR = Path("docs/variable_inventories")


def run_ch_missingness(ch_csv_path: Path) -> pd.DataFrame:
    """Compute column-level missingness and unique value counts."""
    df = pd.read_csv(ch_csv_path, low_memory=False)
    total_rows = len(df)

    records = []
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        records.append({
            "Variable": col,
            "Missing_Count": n_missing,
            "Missing_Percentage": round((n_missing / total_rows) * 100, 2),
            "Non_Missing_Count": total_rows - n_missing,
            "Unique_Values": int(df[col].nunique(dropna=True)),
        })

    report = pd.DataFrame(records)
    report.sort_values(by="Missing_Percentage", ascending=False, inplace=True)
    return report


def verify_household_keys(ch_df: pd.DataFrame, hh_df: pd.DataFrame) -> dict:
    """Validate 100% key match between child households and HH.sav via (HH1, HH2)."""
    ch_keys = ch_df[["HH1", "HH2"]].drop_duplicates()
    hh_keys = hh_df[["HH1", "HH2"]]

    hh_has_duplicates = hh_keys.duplicated().any()
    merged = ch_keys.merge(hh_keys, on=["HH1", "HH2"], how="inner")
    match_rate = (len(merged) / len(ch_keys)) * 100 if len(ch_keys) > 0 else 0.0

    return {
        "distinct_child_households": len(ch_keys),
        "hh_keys_unique": not hh_has_duplicates,
        "matched_households": len(merged),
        "match_rate": round(match_rate, 2),
    }


def main() -> None:
    ch_validated_path = PROCESSED_DIR / "LeadLens_CH_Validated.csv"
    if ch_validated_path.exists():
        print(f"Auditing missingness on validated cohort: {ch_validated_path}")
        missing_df = run_ch_missingness(ch_validated_path)
        out_missing = DOCS_DIR / "LeadLens_CH_Missing_Analysis.csv"
        missing_df.to_csv(out_missing, index=False, encoding="utf-8-sig")
        print(f"Updated missingness analysis saved to {out_missing}")
    else:
        print(f"Note: {ch_validated_path} not found locally. Skipping local CH run.")

    hh_sav_path = SAV_DIR / "hh.sav"
    if hh_sav_path.exists() and ch_validated_path.exists():
        ch_df = pd.read_csv(ch_validated_path, usecols=["HH1", "HH2"], low_memory=False)
        hh_df, hh_meta = pyreadstat.read_sav(str(hh_sav_path), usecols=["HH1", "HH2"])
        stats = verify_household_keys(ch_df, hh_df)
        print(f"Household linkage stats: {stats}")
    else:
        print(f"SPSS hh.sav or validated cohort not present locally; key audit skipped in dry-run.")


if __name__ == "__main__":
    main()

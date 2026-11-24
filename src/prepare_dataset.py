"""Milestone 1: Load Bangladesh MICS7 SPSS files, build variable inventories,
and construct the initial eligible child blood-lead cohort.

Raw .sav files and the output child-level CSV remain strictly local and are
excluded from version control via .gitignore.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat


# Set MICS_SAV_DIR in your environment, or place .sav files in data/raw/
SAV_DIR = Path(os.environ.get("MICS_SAV_DIR", "data/raw"))
INVENTORY_DIR = Path("docs/variable_inventories")
PROCESSED_DIR = Path("data/processed")


def create_inventory(df: pd.DataFrame, metadata, dataset_name: str) -> pd.DataFrame:
    """Extract variable-level metadata and aggregate missingness (no row data)."""
    column_labels = metadata.column_labels
    value_labels = metadata.variable_value_labels
    missing_ranges = getattr(metadata, "missing_ranges", {})

    rows = []
    for i, col in enumerate(df.columns):
        description = column_labels[i] if i < len(column_labels) else ""
        categories = value_labels.get(col, {})
        category_info = " | ".join(f"{k}: {v}" for k, v in categories.items())
        missing_info = missing_ranges.get(col, [])

        rows.append({
            "Dataset": dataset_name,
            "Variable": col,
            "Description": description,
            "Data_Type": str(df[col].dtype),
            "Unique_Values": int(df[col].nunique(dropna=True)),
            "Missing_Count": int(df[col].isna().sum()),
            "Missing_Percentage": round(float(df[col].isna().mean() * 100), 2),
            "Value_Labels": category_info,
            "SPSS_Missing_Definitions": str(missing_info),
        })

    inv_df = pd.DataFrame(rows)
    inv_df.to_csv(
        INVENTORY_DIR / f"{dataset_name}_inventory.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return inv_df


def build_initial_cohort(ch_df: pd.DataFrame) -> pd.DataFrame:
    """Filter eligible children with valid lab status and numeric blood lead."""
    lead_numeric = pd.to_numeric(ch_df["C_Lead"], errors="coerce")

    selected = ch_df["UF8A"] == 1
    valid_lab = ch_df["C_labresult"] == 1
    valid_measurement = lead_numeric.notna()

    cohort = ch_df.loc[selected & valid_lab & valid_measurement].copy()
    cohort["C_Lead"] = lead_numeric.loc[cohort.index]
    cohort["Elevated_Lead"] = (cohort["C_Lead"] >= 5).astype(int)
    return cohort


def main() -> None:
    INVENTORY_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    required_files = ["ch.sav", "hh.sav", "bh.sav"]
    missing_files = [f for f in required_files if not (SAV_DIR / f).exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing SPSS files in {SAV_DIR.resolve()}: {missing_files}. "
            "Place MICS7 .sav files in data/raw/ or set MICS_SAV_DIR."
        )

    ch_df, ch_meta = pyreadstat.read_sav(str(SAV_DIR / "ch.sav"))
    hh_df, hh_meta = pyreadstat.read_sav(str(SAV_DIR / "hh.sav"))
    bh_df, bh_meta = pyreadstat.read_sav(str(SAV_DIR / "bh.sav"))

    print(f"Loaded datasets -> CH: {ch_df.shape}, HH: {hh_df.shape}, BH: {bh_df.shape}")

    ch_inv = create_inventory(ch_df, ch_meta, "CH")
    hh_inv = create_inventory(hh_df, hh_meta, "HH")
    bh_inv = create_inventory(bh_df, bh_meta, "BH")

    all_inv = pd.concat([ch_inv, hh_inv, bh_inv], ignore_index=True)
    all_inv.to_csv(
        INVENTORY_DIR / "ALL_variables_inventory.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Saved variable inventories ({len(all_inv)} total variables) to {INVENTORY_DIR}")

    cohort = build_initial_cohort(ch_df)
    dup_count = int(cohort.duplicated(subset=["HH1", "HH2", "LN"]).sum())

    print("\n--- Initial Child Blood-Lead Cohort Summary ---")
    print(f"Total observations: {len(cohort)} | Total features: {cohort.shape[1]}")
    print(f"Duplicate child IDs (HH1, HH2, LN): {dup_count}")
    print("Elevated_Lead (>= 5 ug/dL) distribution (%):")
    print((cohort["Elevated_Lead"].value_counts(normalize=True) * 100).round(2))

    if "chblweight" in cohort.columns:
        weights = pd.to_numeric(cohort["chblweight"], errors="coerce")
        valid_w = weights.notna() & (weights > 0)
        weighted_prev = np.average(
            cohort.loc[valid_w, "Elevated_Lead"],
            weights=weights[valid_w],
        ) * 100
        print(f"Survey-weighted elevated lead prevalence: {weighted_prev:.2f}% (n={int(valid_w.sum())})")

    out_path = PROCESSED_DIR / "LeadLens_CH_Initial.csv"
    cohort.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nLocal cohort saved (gitignored): {out_path}")


if __name__ == "__main__":
    main()

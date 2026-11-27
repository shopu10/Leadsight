"""Train, validate, and export the finalized 15-question LeadSight production model bundle."""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyreadstat
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_DIR = Path(r"C:\Users\shopu\LeadSight")
DEFAULT_SAV_DIR = Path(
    r"C:\Users\shopu\Downloads\BGD_2025_MICS7_v01_M\BGD_2025_MICS7_v01_M"
    r"\BGD_2025_MICS7_Datasets\Bangladesh MICS7 Datasets"
    r"\Bangladesh MICS7 Datasets\Bangladesh MICS7 SPSS Datasets"
)
DATA_PATH = REPO_DIR / "data" / "processed" / "LeadSight_Modeling_Dataset.csv"
MODEL_OUT = REPO_DIR / "models" / "leadsight_parent_model.joblib"
SUMMARY_OUT = REPO_DIR / "docs" / "production_model_summary.json"


def engineer_production_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    lt17_cols = [
        "nearby_make_battery", "nearby_melt_battery", "nearby_break_battery",
        "nearby_jewelry", "nearby_ewaste", "nearby_auto_repair",
    ]
    lt18_cols = [
        "contact_make_battery", "contact_melt_battery", "contact_break_battery",
        "contact_jewelry", "contact_ewaste", "contact_auto_repair",
    ]

    d["nearby_battery_yes"] = (
        d[["nearby_make_battery", "nearby_melt_battery", "nearby_break_battery"]] == "Yes"
    ).any(axis=1).astype(int)
    d["nearby_ewaste_yes"] = (d["nearby_ewaste"] == "Yes").astype(int)
    d["nearby_auto_jewelry_yes"] = (
        d[["nearby_auto_repair", "nearby_jewelry"]] == "Yes"
    ).any(axis=1).astype(int)
    d["nearby_unsure_any"] = (d[lt17_cols] == "Not_Sure").any(axis=1).astype(int)

    # Consolidated household/regular-contact high-risk occupation flag (eliminates sparse overlap)
    d["contact_high_risk_job"] = (d[lt18_cols] == "Yes").any(axis=1).astype(int)
    d["contact_unsure_any"] = (d[lt18_cols] == "Not_Sure").any(axis=1).astype(int)
    return d


def weighted_ece(y, p, w, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    total_w = w.sum()
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0 or w[m].sum() == 0:
            continue
        wb = w[m]
        ece += (wb.sum() / total_w) * abs(np.sum(wb * y[m]) / wb.sum() - np.sum(wb * p[m]) / wb.sum())
    return float(ece)


def clean_label_map(raw_dict):
    out = {}
    for k, v in raw_dict.items():
        try:
            key_str = str(int(float(k)))
        except (ValueError, TypeError):
            key_str = str(k)
        out[key_str] = str(v).strip().title()
    return out


def main():
    sav_dir = Path(os.environ.get("MICS_SAV_DIR", str(DEFAULT_SAV_DIR)))
    _, ch_meta = pyreadstat.read_sav(str(sav_dir / "ch.sav"), metadataonly=True)

    df = pd.read_csv(
        DATA_PATH,
        dtype={"division_code": str, "district_code": str},
        low_memory=False,
    )
    df = engineer_production_features(df)

    # Feature groups: scale ONLY continuous/ordinal; keep 0/1 flags unscaled for clean log-odds
    num_scaled_cols = [
        "child_age_months",
        "mother_education_ord",
        "sleeping_rooms_capped",
        "pica_freq_ord",
        "contact_surma_ord",
    ]
    bin_unscaled_cols = [
        "nearby_battery_yes",
        "nearby_ewaste_yes",
        "nearby_auto_jewelry_yes",
        "nearby_unsure_any",
        "contact_high_risk_job",
        "contact_unsure_any",
    ]
    cat_cols = [
        "child_sex",
        "area_type",
        "division_code",
        "district_code",
        "floor_group",
        "wall_group",
        "interior_paint",
        "water_source_group",
    ]

    div_codes = sorted(df["division_code"].dropna().unique().tolist(), key=lambda x: int(x))
    dist_codes = sorted(df["district_code"].dropna().unique().tolist(), key=lambda x: int(x))

    cat_categories = [
        ["Female", "Male"],
        ["Rural", "Urban"],
        div_codes,
        dist_codes,
        ["Natural_Earth_Dung", "Finished_Cement", "Finished_Tiles_Other", "Rudimentary_Other"],
        ["Natural_Mud_Bamboo", "Tin", "Bricks", "Cement_Concrete", "Other_Wood"],
        ["No_Paint", "Good_Paint", "Peeling_Paint"],
        ["Tubewell_Shallow", "Tubewell_Deep", "Piped_Dwelling_Yard", "Piped_Shared_Public", "Other_Water_Source"],
    ]

    def build_pipeline(C=0.25):
        prep = ColumnTransformer([
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
            ]), num_scaled_cols),
            ("bin", SimpleImputer(strategy="most_frequent"), bin_unscaled_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(
                    categories=cat_categories,
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                )),
            ]), cat_cols),
        ], remainder="drop")
        return Pipeline([
            ("prep", prep),
            ("clf", LogisticRegression(C=C, max_iter=1000, random_state=42)),
        ])

    feature_cols = num_scaled_cols + bin_unscaled_cols + cat_cols
    X = df[feature_cols].copy()
    y = df["Elevated_Lead"].astype(int).values
    groups = df["PSU"].values
    w = pd.to_numeric(df["chblweight"], errors="coerce").fillna(0.0).clip(lower=0.0).values
    w_norm = w / np.mean(w)

    # 1. 5-fold PSU-grouped CV evaluation
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(df), dtype=float)
    fold_aucs = []
    for tr, va in sgkf.split(X, y, groups):
        p = build_pipeline(C=0.25)
        p.fit(X.iloc[tr], y[tr], clf__sample_weight=w_norm[tr])
        preds = p.predict_proba(X.iloc[va])[:, 1]
        oof[va] = preds
        fold_aucs.append(roc_auc_score(y[va], preds))

    metrics = {
        "n_children": int(len(df)),
        "age_range_months": [12, 59],
        "n_parent_questions": 15,
        "unweighted_prevalence": round(float(y.mean()), 4),
        "weighted_prevalence": round(float(np.sum(w * y) / w.sum()), 4),
        "mean_predicted_weighted": round(float(np.sum(w * oof) / w.sum()), 4),
        "roc_auc_unweighted": round(float(roc_auc_score(y, oof)), 4),
        "roc_auc_fold_sd": round(float(np.std(fold_aucs)), 4),
        "roc_auc_weighted": round(float(roc_auc_score(y, oof, sample_weight=w)), 4),
        "pr_auc_unweighted": round(float(average_precision_score(y, oof)), 4),
        "pr_auc_weighted": round(float(average_precision_score(y, oof, sample_weight=w)), 4),
        "brier_weighted": round(float(brier_score_loss(y, oof, sample_weight=w)), 4),
        "ece_weighted": round(weighted_ece(y, oof, w), 4),
    }

    # 2. Empirically validated 4-tier risk stratification on OOF predictions
    tier_bins = [0.0, 0.25, 0.45, 0.65, 1.01]
    tier_labels = ["Low", "Moderate", "High", "Very High"]
    tier_series = pd.cut(oof, bins=tier_bins, labels=tier_labels, right=False)

    tier_summary = []
    for lbl in tier_labels:
        m = np.asarray(tier_series == lbl)
        n_t = int(m.sum())
        unw_r = float(y[m].mean() * 100) if n_t else 0.0
        wtd_r = float(np.sum(w[m] * y[m]) / w[m].sum() * 100) if n_t and w[m].sum() > 0 else 0.0
        share_w = float(w[m].sum() / w.sum() * 100) if n_t else 0.0
        tier_summary.append({
            "tier": lbl,
            "n_children": n_t,
            "weighted_pop_share_pct": round(share_w, 2),
            "observed_unweighted_pct": round(unw_r, 2),
            "observed_weighted_pct": round(wtd_r, 2),
        })

    # 3. Fit finalized production model on full 10,622 survey-weighted cohort
    final_pipe = build_pipeline(C=0.25)
    final_pipe.fit(X, y, clf__sample_weight=w_norm)

    feat_names = final_pipe.named_steps["prep"].get_feature_names_out()
    coefs = final_pipe.named_steps["clf"].coef_[0]
    coef_df = pd.DataFrame({
        "Feature": feat_names,
        "LogOdds": np.round(coefs, 4),
        "OddsRatio": np.round(np.exp(coefs), 3),
    })

    # Compute weighted population mean transformed vector for exact local explanations
    X_trans = final_pipe.named_steps["prep"].transform(X)
    baseline_trans_mean = np.average(X_trans, axis=0, weights=w_norm)

    # 4. Geographic display labels & Division -> District lookup
    div_labels = clean_label_map(ch_meta.variable_value_labels.get("HH7", {}))
    dist_labels = clean_label_map(ch_meta.variable_value_labels.get("HH7A", {}))

    div_to_dists = {}
    for dv in div_codes:
        sub_dists = sorted(
            df.loc[df["division_code"] == dv, "district_code"].dropna().unique().tolist(),
            key=lambda x: dist_labels.get(x, x),
        )
        div_to_dists[dv] = [{"code": dc, "name": dist_labels.get(dc, dc)} for dc in sub_dists]

    # Save deployable bundle (contains zero raw survey rows)
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": final_pipe,
        "feature_columns": feature_cols,
        "num_scaled_cols": num_scaled_cols,
        "bin_unscaled_cols": bin_unscaled_cols,
        "cat_cols": cat_cols,
        "transformed_feature_names": feat_names.tolist(),
        "baseline_transformed_mean": baseline_trans_mean.tolist(),
        "division_labels": {dv: div_labels.get(dv, dv) for dv in div_codes},
        "district_labels": {dc: dist_labels.get(dc, dc) for dc in dist_codes},
        "division_to_districts": div_to_dists,
        "risk_tiers": {
            "cutoffs": [0.25, 0.45, 0.65],
            "labels": tier_labels,
            "oof_validation": tier_summary,
        },
        "metrics": metrics,
    }
    joblib.dump(bundle, MODEL_OUT)

    summary_payload = {
        "metrics": metrics,
        "oof_risk_tiers": tier_summary,
        "non_geographic_odds_ratios": coef_df[
            ~coef_df["Feature"].str.contains("district_code|division_code")
        ].sort_values("LogOdds", ascending=False).to_dict(orient="records"),
    }
    SUMMARY_OUT.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("=== FINAL 15-QUESTION PRODUCTION MODEL (5-FOLD PSU CV, SURVEY-WEIGHTED) ===")
    for k, v in metrics.items():
        print(f"{k:26s}: {v}")

    print("\n=== EMPIRICAL 4-TIER RISK STRATIFICATION (OUT-OF-FOLD VALIDATION) ===")
    print(pd.DataFrame(tier_summary).to_string(index=False))

    print("\n=== FINAL NON-GEOGRAPHIC ODDS RATIOS (UNSCALED BINARIES + SCALED ORDINALS) ===")
    non_geo = coef_df[~coef_df["Feature"].str.contains("district_code|division_code")].sort_values("LogOdds", ascending=False)
    print(non_geo.to_string(index=False))

    print(f"\nSaved model bundle:  {MODEL_OUT} ({MODEL_OUT.stat().st_size / 1024:.1f} KB)")
    print(f"Saved model summary: {SUMMARY_OUT}")


if __name__ == "__main__":
    main()

"""Cluster-grouped 5-fold CV benchmark for LeadSight (Parent vs Geo vs Soil models)."""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

REPO_DIR = Path(r"C:\Users\shopu\LeadSight")
DATA_PATH = REPO_DIR / "data" / "processed" / "LeadSight_Modeling_Dataset.csv"
SCHEMA_PATH = REPO_DIR / "docs" / "feature_schema.json"
OUT_CSV = REPO_DIR / "docs" / "benchmark_results.csv"


def build_preprocessor(num_cols, cat_cols, scale_num=False):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_num:
        num_steps.append(("scaler", StandardScaler()))

    cat_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline(num_steps), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline(cat_steps), cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_models():
    return [
        ("LogisticRegression", LogisticRegression(C=0.5, max_iter=1000, random_state=42), True),
        ("RandomForest", RandomForestClassifier(
            n_estimators=250, min_samples_leaf=10, max_features="sqrt", n_jobs=-1, random_state=42
        ), False),
        ("HistGradientBoosting", HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=30, l2_regularization=1.0, random_state=42
        ), False),
        ("MLP_NeuralNet", MLPClassifier(
            hidden_layer_sizes=(64, 32), alpha=0.05, early_stopping=True,
            max_iter=100, random_state=42
        ), True),
    ]


def evaluate_tier(df_sub, num_cols, cat_cols, target_col, tier_name):
    df_sub = df_sub.reset_index(drop=True)
    feat_cols = num_cols + cat_cols
    X = df_sub[feat_cols].copy()
    y = df_sub[target_col].astype(int).values
    groups = df_sub["PSU"].values
    weights = pd.to_numeric(df_sub["chblweight"], errors="coerce").fillna(0.0).clip(lower=0.0).values

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(sgkf.split(X, y, groups))

    rows = []
    for model_name, estimator, scale_num in get_models():
        oof_prob = np.zeros(len(df_sub), dtype=float)
        fold_aucs = []

        for tr_idx, va_idx in splits:
            pipe = Pipeline([
                ("prep", build_preprocessor(num_cols, cat_cols, scale_num=scale_num)),
                ("clf", estimator),
            ])
            pipe.fit(X.iloc[tr_idx], y[tr_idx])
            p = pipe.predict_proba(X.iloc[va_idx])[:, 1]
            oof_prob[va_idx] = p
            fold_aucs.append(roc_auc_score(y[va_idx], p))

        rows.append({
            "Tier": tier_name,
            "Target": target_col,
            "Model": model_name,
            "n": len(df_sub),
            "n_features": len(feat_cols),
            "ROC_AUC": round(roc_auc_score(y, oof_prob), 4),
            "ROC_AUC_SD": round(float(np.std(fold_aucs)), 4),
            "ROC_AUC_Wtd": round(roc_auc_score(y, oof_prob, sample_weight=weights), 4),
            "PR_AUC": round(average_precision_score(y, oof_prob), 4),
            "PR_AUC_Wtd": round(average_precision_score(y, oof_prob, sample_weight=weights), 4),
            "Brier": round(brier_score_loss(y, oof_prob), 4),
            "Brier_Wtd": round(brier_score_loss(y, oof_prob, sample_weight=weights), 4),
        })
    return rows


def main():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    df = pd.read_csv(
        DATA_PATH,
        dtype={"division_code": str, "district_code": str},
        low_memory=False,
    )

    p_num = schema["parent_features"]["numeric"]
    p_cat = schema["parent_features"]["categorical"]
    soil_num = schema["soil_env_features"]

    geo_cat = ["area_type", "division_code", "district_code"]
    p_cat_no_water = [c for c in p_cat if c != "water_source_group"]

    soil_mask = df[soil_num].notna().all(axis=1)
    df_soil = df.loc[soil_mask].copy()

    tiers = [
        ("T1_Geo_Only", df, [], geo_cat, "Elevated_Lead"),
        ("T2_Parent_No_Water", df, p_num, p_cat_no_water, "Elevated_Lead"),
        ("T3_Parent_Full", df, p_num, p_cat, "Elevated_Lead"),
        ("T3b_Parent_SoilCohort", df_soil, p_num, p_cat, "Elevated_Lead"),
        ("T4_Parent_Plus_Soil", df_soil, p_num + soil_num, p_cat, "Elevated_Lead"),
        ("T5_Parent_Sens_3_5", df, p_num, p_cat, "Elevated_Lead_3_5"),
    ]

    all_rows = []
    for tier_name, dsub, n_cols, c_cols, tgt in tiers:
        print(f"Running {tier_name} (n={len(dsub)}, target={tgt})...")
        all_rows.extend(evaluate_tier(dsub, n_cols, c_cols, tgt, tier_name))

    res = pd.DataFrame(all_rows)
    res.to_csv(OUT_CSV, index=False)

    print("\n=== PRIMARY BENCHMARK (Elevated_Lead >= 5 ug/dL) ===")
    prim = res[res["Target"] == "Elevated_Lead"]
    print(prim.drop(columns=["Target"]).to_string(index=False))

    print("\n=== SECONDARY SENSITIVITY (Elevated_Lead_3_5 >= 3.5 ug/dL) ===")
    sec = res[res["Target"] == "Elevated_Lead_3_5"]
    print(sec.drop(columns=["Target"]).to_string(index=False))

    print("\n=== BEST MODEL PER TIER (BY SURVEY-WEIGHTED ROC-AUC) ===")
    best = res.sort_values("ROC_AUC_Wtd", ascending=False).groupby("Tier", sort=False).first().reset_index()
    print(best[["Tier", "Model", "n", "ROC_AUC", "ROC_AUC_Wtd", "PR_AUC_Wtd", "Brier_Wtd"]].to_string(index=False))
    print(f"\nSaved full benchmark table to: {OUT_CSV}")


if __name__ == "__main__":
    main()

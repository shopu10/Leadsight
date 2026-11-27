"""Audit LogisticRegression: weighted vs unweighted calibration, C tuning, within-district AUC, and odds ratios."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_DIR = Path(r"C:\Users\shopu\LeadSight")
DATA_PATH = REPO_DIR / "data" / "processed" / "LeadSight_Modeling_Dataset.csv"
SCHEMA_PATH = REPO_DIR / "docs" / "feature_schema.json"


def make_prep(num_cols, cat_cols):
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
        ]), cat_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def cv_oof(df, num_cols, cat_cols, C=0.5, use_weights=False):
    X = df[num_cols + cat_cols].copy()
    y = df["Elevated_Lead"].astype(int).values
    groups = df["PSU"].values
    w = pd.to_numeric(df["chblweight"], errors="coerce").fillna(0.0).clip(lower=0.0).values
    w_norm = w / np.mean(w)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(df), dtype=float)

    for tr, va in sgkf.split(X, y, groups):
        pipe = Pipeline([
            ("prep", make_prep(num_cols, cat_cols)),
            ("clf", LogisticRegression(C=C, max_iter=1000, random_state=42)),
        ])
        if use_weights:
            pipe.fit(X.iloc[tr], y[tr], clf__sample_weight=w_norm[tr])
        else:
            pipe.fit(X.iloc[tr], y[tr])
        oof[va] = pipe.predict_proba(X.iloc[va])[:, 1]
    return oof


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
        acc = np.sum(wb * y[m]) / wb.sum()
        conf = np.sum(wb * p[m]) / wb.sum()
        ece += (wb.sum() / total_w) * abs(acc - conf)
    return ece


def within_district_auc(df, prob_col):
    aucs, ns = [], []
    for _, g in df.groupby("district_code"):
        if len(g) >= 50 and g["Elevated_Lead"].nunique() == 2:
            aucs.append(roc_auc_score(g["Elevated_Lead"], g[prob_col]))
            ns.append(len(g))
    return float(np.average(aucs, weights=ns)), len(aucs)


def main():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    df = pd.read_csv(DATA_PATH, dtype={"division_code": str, "district_code": str}, low_memory=False)

    p_num = schema["parent_features"]["numeric"]
    p_cat = schema["parent_features"]["categorical"]
    y = df["Elevated_Lead"].astype(int).values
    w = pd.to_numeric(df["chblweight"], errors="coerce").fillna(0.0).clip(lower=0.0).values
    w_prev = float(np.sum(w * y) / w.sum())

    print(f"Cohort n={len(df)} | Unwtd prevalence={y.mean()*100:.2f}% | Survey-Wtd prevalence={w_prev*100:.2f}%\n")

    print("=== 1. C TUNING & UNWEIGHTED VS SURVEY-WEIGHTED TRAINING (T3_Parent_Full) ===")
    rows = []
    best_oof_w = None
    best_oof_u = None
    for use_w in [False, True]:
        for C in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]:
            oof = cv_oof(df, p_num, p_cat, C=C, use_weights=use_w)
            w_mean_p = float(np.sum(w * oof) / w.sum())
            rows.append({
                "Fit_Wtd": use_w,
                "C": C,
                "AUC_Unwtd": round(roc_auc_score(y, oof), 4),
                "AUC_Wtd": round(roc_auc_score(y, oof, sample_weight=w), 4),
                "PR_Wtd": round(average_precision_score(y, oof, sample_weight=w), 4),
                "Brier_Wtd": round(brier_score_loss(y, oof, sample_weight=w), 4),
                "ECE_Wtd": round(weighted_ece(y, oof, w), 4),
                "Mean_Pred_Wtd%": round(w_mean_p * 100, 2),
            })
            if not use_w and C == 0.25:
                best_oof_u = oof
            if use_w and C == 0.25:
                best_oof_w = oof
    print(pd.DataFrame(rows).to_string(index=False))

    # 2. Within-district ranking proof
    df["oof_geo"] = cv_oof(df, [], ["area_type", "division_code", "district_code"], C=0.25, use_weights=True)
    df["oof_parent"] = best_oof_w
    g_auc, n_d = within_district_auc(df, "oof_geo")
    p_auc, _ = within_district_auc(df, "oof_parent")
    print(f"\n=== 2. WITHIN-DISTRICT DISCRIMINATION ({n_d} districts with n>=50) ===")
    print(f"Geo-Only Mean Within-District AUC:    {g_auc:.4f}")
    print(f"Parent-Full Mean Within-District AUC: {p_auc:.4f} (Lift: {p_auc - g_auc:+.4f})")

    # 3. Fit full weighted model (C=0.25) and inspect all NON-GEOGRAPHIC odds ratios
    X_all = df[p_num + p_cat].copy()
    w_norm = w / np.mean(w)
    pipe = Pipeline([
        ("prep", make_prep(p_num, p_cat)),
        ("clf", LogisticRegression(C=0.25, max_iter=1000, random_state=42)),
    ])
    pipe.fit(X_all, y, clf__sample_weight=w_norm)
    feat_names = pipe.named_steps["prep"].get_feature_names_out()
    coefs = pipe.named_steps["clf"].coef_[0]

    coef_df = pd.DataFrame({
        "Feature": feat_names,
        "LogOdds": np.round(coefs, 4),
        "OddsRatio": np.round(np.exp(coefs), 3),
    })
    non_geo = coef_df[~coef_df["Feature"].str.contains("district_code|division_code")].copy()
    non_geo = non_geo.sort_values("LogOdds", key=abs, ascending=False)
    print("\n=== 3. NON-GEOGRAPHIC ODDS RATIOS (SURVEY-WEIGHTED LOGREG, C=0.25) ===")
    print(non_geo.to_string(index=False))


if __name__ == "__main__":
    main()

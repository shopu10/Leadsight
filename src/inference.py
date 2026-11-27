"""Production inference, exact additive log-odds attribution, and tailored guidance for LeadSight."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = REPO_DIR / "models" / "leadsight_parent_model.joblib"


def load_bundle(model_path: Path = DEFAULT_MODEL_PATH) -> dict:
    return joblib.load(model_path)


def _group_label(transformed_name: str) -> str:
    if any(k in transformed_name for k in ("division_code", "district_code", "area_type")):
        return "Location (Division, District & Urban/Rural)"
    if "water_source_group" in transformed_name:
        return "Drinking Water Source"
    if "floor_group" in transformed_name:
        return "Home Floor Material"
    if "wall_group" in transformed_name:
        return "Exterior Wall Material"
    if "interior_paint" in transformed_name:
        return "Interior Paint Condition"
    if "mother_education_ord" in transformed_name:
        return "Mother's Education Level"
    if "sleeping_rooms_capped" in transformed_name:
        return "Household Crowding (Sleeping Rooms)"
    if "child_age_months" in transformed_name:
        return "Child's Age (12-59 months)"
    if "child_sex" in transformed_name:
        return "Child's Sex"
    if "contact_surma_ord" in transformed_name:
        return "Household Surma (Kohl) Use"
    if "pica_freq_ord" in transformed_name:
        return "Eating Dirt, Clay, or Paint Chips (Pica)"
    if any(k in transformed_name for k in ("nearby_battery", "nearby_ewaste", "nearby_auto_jewelry", "nearby_unsure")):
        return "Nearby Lead-Risk Workshops"
    if any(k in transformed_name for k in ("contact_high_risk_job", "contact_unsure")):
        return "Household Contact Job Exposure"
    return transformed_name


def classify_tier(prob: float, risk_tiers: dict) -> dict:
    cutoffs = risk_tiers["cutoffs"]
    labels = risk_tiers["labels"]
    idx = 0
    for c in cutoffs:
        if prob >= c:
            idx += 1
    tier_name = labels[min(idx, len(labels) - 1)]
    oof_match = next(
        (row for row in risk_tiers["oof_validation"] if row["tier"] == tier_name),
        {},
    )
    return {
        "tier": tier_name,
        "oof_observed_weighted_pct": oof_match.get("observed_weighted_pct"),
        "oof_n_children": oof_match.get("n_children"),
    }


def build_recommendations(inputs: dict, tier: str) -> list[dict]:
    """Generate practical, non-diagnostic public-health guidance tied directly to user inputs."""
    recs = []

    # 1. Blood testing guidance proportional to risk tier
    if tier in ("Very High", "High"):
        recs.append({
            "priority": "High Priority",
            "title": "Consult a pediatrician about a venous blood lead test",
            "detail": (
                "Because your child's estimated risk falls in the higher tier, discuss a laboratory "
                "blood lead level (BLL) test with a qualified pediatrician or public-health diagnostic center. "
                "LeadSight is a screening estimate, not a medical diagnosis."
            ),
        })
    elif tier == "Moderate":
        recs.append({
            "priority": "Recommended",
            "title": "Ask about lead screening during your next routine child check-up",
            "detail": (
                "Share any nearby industrial, paint, or household exposures with your doctor and focus on "
                "reducing everyday dust and ingestion pathways below."
            ),
        })
    else:
        recs.append({
            "priority": "Preventive",
            "title": "Maintain everyday lead-safe habits",
            "detail": (
                "Your child's estimated probability is below the national average, though no blood lead level "
                "is completely safe. Keep routine handwashing and dust-control habits in place."
            ),
        })

    # 2. Modifiable pathway actions
    if inputs.get("interior_paint") == "Peeling_Paint":
        recs.append({
            "priority": "Actionable Home Step",
            "title": "Control peeling or chipping paint dust safely",
            "detail": (
                "Avoid dry-scraping or sanding peeling walls while children are present. Wipe floors and "
                "windowsills regularly with a damp cloth or wet mop, and keep children away from flaking paint."
            ),
        })

    if inputs.get("pica_freq_ord", 1) >= 2:
        recs.append({
            "priority": "Actionable Behavior Step",
            "title": "Reduce hand-to-mouth soil, clay, and dust ingestion",
            "detail": (
                "Wash your child's hands and face before eating and after playing, keep pacifiers/toys clean, "
                "and prioritize iron-, calcium-, and vitamin C-rich foods (leafy greens, dairy, lentils, citrus) "
                "which help reduce lead absorption."
            ),
        })

    if inputs.get("contact_surma_ord", 1) >= 2:
        recs.append({
            "priority": "Actionable Household Step",
            "title": "Avoid traditional surma/kohl around young children",
            "detail": (
                "Many unregulated surma/kohl products in South Asia contain high concentrations of lead sulfide. "
                "Avoid applying surma to children and wash hands carefully after handling traditional cosmetics."
            ),
        })

    if inputs.get("contact_high_risk_job", 0) == 1:
        recs.append({
            "priority": "Actionable Take-Home Step",
            "title": "Block take-home lead dust from high-risk work",
            "detail": (
                "Household members working with batteries, e-waste, auto repair, or metal/jewelry should "
                "change clothes and shoes before entering living areas, wash work clothes separately, and shower "
                "before holding the child."
            ),
        })

    if any(inputs.get(k, 0) == 1 for k in ("nearby_battery_yes", "nearby_ewaste_yes", "nearby_auto_jewelry_yes")):
        recs.append({
            "priority": "Actionable Environment Step",
            "title": "Minimize outdoor workshop dust tracking indoors",
            "detail": (
                "Keep shoes outside the living space, wet-mop floors instead of dry-sweeping, keep food and "
                "drinking water covered, and keep children away from nearby battery, e-waste, or repair sites."
            ),
        })

    if inputs.get("water_source_group") in ("Piped_Dwelling_Yard", "Piped_Shared_Public"):
        recs.append({
            "priority": "Water Safety Habit",
            "title": "Flush standing piped tap water before drinking or cooking",
            "detail": (
                "If municipal piped water sits in household plumbing overnight, let cold water run briefly "
                "before collecting drinking or cooking water—especially in older buildings."
            ),
        })

    return recs


def predict_and_explain(inputs: dict, bundle: dict) -> dict:
    """Return calibrated probability, risk tier, exact grouped log-odds drivers, and action plan."""
    age = float(inputs["child_age_months"])
    if not (12.0 <= age <= 59.0):
        raise ValueError("LeadSight is validated strictly for children aged 12 to 59 months.")

    feature_cols = bundle["feature_columns"]
    row_df = pd.DataFrame([{c: inputs[c] for c in feature_cols}])

    pipe = bundle["pipeline"]
    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]

    prob = float(pipe.predict_proba(row_df)[0, 1])
    z = prep.transform(row_df)[0]
    baseline = np.asarray(bundle["baseline_transformed_mean"], dtype=float)
    coefs = clf.coef_[0]
    intercept = float(clf.intercept_[0])

    # Exact additive decomposition relative to nationally weighted MICS7 population average
    raw_contribs = (z - baseline) * coefs
    baseline_logit = float(intercept + np.dot(baseline, coefs))
    model_logit = float(intercept + np.dot(z, coefs))

    grouped = {}
    for fname, cval in zip(bundle["transformed_feature_names"], raw_contribs):
        grp = _group_label(fname)
        grouped[grp] = grouped.get(grp, 0.0) + float(cval)

    contrib_df = (
        pd.DataFrame([
            {
                "Factor": k,
                "LogOdds_Shift": round(v, 4),
                "Odds_Multiplier": round(float(np.exp(v)), 3),
                "Direction": "Increases Risk" if v > 0.02 else ("Protective" if v < -0.02 else "Minimal Effect"),
            }
            for k, v in grouped.items()
        ])
        .sort_values("LogOdds_Shift", key=abs, ascending=False)
        .reset_index(drop=True)
    )

    tier_info = classify_tier(prob, bundle["risk_tiers"])
    recommendations = build_recommendations(inputs, tier_info["tier"])

    return {
        "probability": prob,
        "probability_pct": round(prob * 100.0, 1),
        "national_weighted_avg_pct": round(bundle["metrics"]["weighted_prevalence"] * 100.0, 1),
        "tier": tier_info["tier"],
        "oof_observed_weighted_pct": tier_info["oof_observed_weighted_pct"],
        "oof_n_children": tier_info["oof_n_children"],
        "baseline_logit": baseline_logit,
        "model_logit": model_logit,
        "decomposition_error": abs((baseline_logit + sum(grouped.values())) - model_logit),
        "contributions": contrib_df,
        "recommendations": recommendations,
    }

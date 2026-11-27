"""LeadSight — Parent-Friendly Childhood Lead Exposure Risk Assessment System for Bangladesh."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.inference import load_bundle, predict_and_explain

REPO_DIR = Path(__file__).resolve().parent
BENCHMARK_CSV = REPO_DIR / "docs" / "benchmark_results.csv"
SUMMARY_JSON = REPO_DIR / "docs" / "production_model_summary.json"

st.set_page_config(
    page_title="LeadSight Bangladesh — Childhood Lead Risk Screener",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource
def get_bundle():
    return load_bundle()


def render_contribution_chart(contrib_df: pd.DataFrame):
    top = contrib_df[contrib_df["LogOdds_Shift"].abs() >= 0.015].head(10).iloc[::-1]
    if top.empty:
        st.info("All inputs sit close to the national MICS 2025 baseline.")
        return

    fig, ax = plt.subplots(figsize=(7.5, max(3.2, 0.42 * len(top))))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in top["LogOdds_Shift"]]
    ax.barh(top["Factor"], top["LogOdds_Shift"], color=colors, edgecolor="none")
    ax.axvline(0.0, color="#9aa0a6", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Log-Odds Shift vs. National Avg  (← Lower Risk | Higher Risk →)", color="#9aa0a6")
    ax.set_title("Exact Model Factor Contributions", color="#9aa0a6", fontweight="bold")
    ax.tick_params(colors="#9aa0a6")
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#9aa0a6")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def main():
    bundle = get_bundle()
    metrics = bundle["metrics"]

    st.title("LeadSight — Childhood Lead Exposure Risk Assessment (Bangladesh)")
    st.caption(
        "Parent-friendly screening tool trained on Bangladesh MICS 2025 (MICS7) data "
        f"(n = {metrics['n_children']:,} validated children aged 12–59 months; threshold ≥ 5 µg/dL)."
    )

    st.warning(
        "**Important Medical Notice:** LeadSight estimates statistical risk from parent-reported household "
        "and community information. It is **not** a medical diagnosis and cannot replace a laboratory blood lead test."
    )

    tab_assess, tab_evidence, tab_method = st.tabs([
        "1. Parent Risk Assessment (15 Questions)",
        "2. Model Evidence & Research Findings",
        "3. Methodology & Scope Limits",
    ])

    # =========================================================================
    # TAB 1: PARENT RISK ASSESSMENT
    # =========================================================================
    with tab_assess:
        st.markdown(
            "Answer **15 simple questions** about your child (aged **1 to 4 years / 12–59 months**), "
            "your home, and your surrounding area. No blood test, lab report, or income document is required."
        )

        col_left, col_right = st.columns([1.15, 1.0], gap="large")

        with col_left:
            st.subheader("A. Child & Caregiver")
            c1, c2 = st.columns(2)
            with c1:
                child_age_months = st.slider(
                    "1. Child's age (months)",
                    min_value=12,
                    max_value=59,
                    value=30,
                    help="LeadSight is validated strictly for children aged 12–59 months (1–4 years).",
                )
            with c2:
                child_sex = st.radio("2. Child's sex", options=["Male", "Female"], horizontal=True)

            edu_map = {
                "No formal education / Pre-primary": 0.0,
                "Primary (Class 1–5)": 1.0,
                "Lower Secondary (Class 6–8)": 2.0,
                "Secondary / SSC (Class 9–10)": 3.0,
                "Higher Secondary / HSC (Class 11–12)": 4.0,
                "Higher (University / Degree)": 5.0,
            }
            edu_choice = st.selectbox("3. Mother's highest education level", list(edu_map.keys()), index=3)

            st.subheader("B. Location & Home Conditions")
            div_labels = bundle["division_labels"]
            div_codes = sorted(div_labels.keys(), key=lambda k: div_labels[k])

            g1, g2, g3 = st.columns(3)
            with g1:
                selected_div = st.selectbox(
                    "4. Division",
                    options=div_codes,
                    format_func=lambda c: div_labels[c],
                    index=div_codes.index("30") if "30" in div_codes else 0,
                )
            with g2:
                dist_options = bundle["division_to_districts"][selected_div]
                dist_codes = [d["code"] for d in dist_options]
                dist_name_map = {d["code"]: d["name"] for d in dist_options}
                selected_dist = st.selectbox(
                    "5. District / City Corp",
                    options=dist_codes,
                    format_func=lambda c: dist_name_map[c],
                )
            with g3:
                area_type = st.radio("6. Area type", options=["Urban", "Rural"], horizontal=True)

            h1, h2 = st.columns(2)
            with h1:
                room_map = {"1 room": 1.0, "2 rooms": 2.0, "3 rooms": 3.0, "4 or more rooms": 4.0}
                room_choice = st.selectbox("7. Rooms used for sleeping", list(room_map.keys()), index=1)

                floor_map = {
                    "Earth / Sand / Mud": "Natural_Earth_Dung",
                    "Cement": "Finished_Cement",
                    "Ceramic Tiles / Finished Wood / Carpet": "Finished_Tiles_Other",
                    "Wood Planks / Other Rudimentary": "Rudimentary_Other",
                }
                floor_choice = st.selectbox("8. Main floor material", list(floor_map.keys()), index=1)

                paint_map = {
                    "No paint on inside walls/doors/windows": "No_Paint",
                    "Painted — good condition (not peeling)": "Good_Paint",
                    "Painted — peeling, chipping, or cracking": "Peeling_Paint",
                }
                paint_choice = st.selectbox("9. Inside wall/door paint condition", list(paint_map.keys()), index=0)

            with h2:
                wall_map = {
                    "Tin / Corrugated Sheet": "Tin",
                    "Cement / Concrete / Stone": "Cement_Concrete",
                    "Bricks": "Bricks",
                    "Mud / Bamboo / Palm / Natural": "Natural_Mud_Bamboo",
                    "Wood Planks / Other": "Other_Wood",
                }
                wall_choice = st.selectbox("10. Main exterior wall material", list(wall_map.keys()), index=1)

                water_map = {
                    "Shallow Tube Well / Borehole": "Tubewell_Shallow",
                    "Deep Tube Well / Borehole": "Tubewell_Deep",
                    "Piped Water into Home or Yard": "Piped_Dwelling_Yard",
                    "Shared / Public Piped Tap": "Piped_Shared_Public",
                    "Dug Well / Rainwater / Kiosk / Surface / Other": "Other_Water_Source",
                }
                water_choice = st.selectbox("11. Main drinking water source", list(water_map.keys()), index=0)

            st.subheader("C. Daily Habits & Nearby Lead Sources (Last 3 Months)")
            freq_map = {
                "Not at all": 1.0,
                "Less than once a week": 2.0,
                "At least once a week": 3.0,
                "Almost every day": 4.0,
            }
            b1, b2 = st.columns(2)
            with b1:
                pica_choice = st.selectbox(
                    "12. Child puts soil, dirt, clay, or paint chips in mouth",
                    list(freq_map.keys()),
                    index=0,
                )
            with b2:
                surma_choice = st.selectbox(
                    "13. Anyone in regular contact with child uses surma (kohl)",
                    list(freq_map.keys()),
                    index=0,
                )

            nearby_options = [
                "Vehicle battery making, melting, or acid-breaking workshop",
                "Electronic waste (e-waste) recycling (phones, computers, scrap)",
                "Automobile repair garage or jewelry making/recycling",
                "Not sure about nearby workshops",
            ]
            nearby_selected = st.multiselect(
                "14. Are any of these activities located near your home?",
                options=nearby_options,
                default=[],
                help="Select all that apply, or leave blank if none.",
            )

            contact_choice = st.radio(
                "15. Does anyone in regular contact with the child work in battery making/recycling, e-waste, auto repair, or jewelry?",
                options=["No", "Yes", "Not sure"],
                horizontal=True,
            )

            run_clicked = st.button("Estimate Childhood Lead Risk", type="primary", use_container_width=True)

        # Build input dict continuously so right pane always reflects current inputs cleanly
        inputs = {
            "child_age_months": float(child_age_months),
            "mother_education_ord": edu_map[edu_choice],
            "sleeping_rooms_capped": room_map[room_choice],
            "pica_freq_ord": freq_map[pica_choice],
            "contact_surma_ord": freq_map[surma_choice],
            "nearby_battery_yes": int(nearby_options[0] in nearby_selected),
            "nearby_ewaste_yes": int(nearby_options[1] in nearby_selected),
            "nearby_auto_jewelry_yes": int(nearby_options[2] in nearby_selected),
            "nearby_unsure_any": int(nearby_options[3] in nearby_selected),
            "contact_high_risk_job": int(contact_choice == "Yes"),
            "contact_unsure_any": int(contact_choice == "Not sure"),
            "child_sex": child_sex,
            "area_type": area_type,
            "division_code": selected_div,
            "district_code": selected_dist,
            "floor_group": floor_map[floor_choice],
            "wall_group": wall_map[wall_choice],
            "interior_paint": paint_map[paint_choice],
            "water_source_group": water_map[water_choice],
        }

        with col_right:
            st.subheader("Risk Assessment Result")
            if run_clicked or "last_result" not in st.session_state:
                st.session_state["last_result"] = predict_and_explain(inputs, bundle)
                st.session_state["last_inputs"] = inputs
            else:
                # Auto-refresh so changing any dropdown keeps the estimate synced
                st.session_state["last_result"] = predict_and_explain(inputs, bundle)
                st.session_state["last_inputs"] = inputs

            res = st.session_state["last_result"]
            tier = res["tier"]

            m1, m2 = st.columns(2)
            m1.metric(
                "Estimated Risk (≥ 5 µg/dL)",
                f"{res['probability_pct']:.1f}%",
                delta=f"{res['probability_pct'] - res['national_weighted_avg_pct']:+.1f}pp vs avg",
                delta_color="inverse",
            )
            m2.metric("Risk Tier", tier)
            st.caption(
                f"**MICS 2025 Tier Benchmark:** {res['oof_observed_weighted_pct']:.1f}% weighted observed "
                f"elevated rate (n = {res['oof_n_children']:,} validated children in this tier)."
            )

            if tier == "Very High":
                st.error(
                    f"**Very High Risk Tier ({res['probability_pct']:.1f}%):** In Bangladesh MICS 2025 validation, "
                    f"**{res['oof_observed_weighted_pct']:.1f}%** of children in this tier had elevated blood lead (≥ 5 µg/dL). "
                    "Strongly consider consulting a doctor for a venous blood lead test."
                )
            elif tier == "High":
                st.warning(
                    f"**High Risk Tier ({res['probability_pct']:.1f}%):** In MICS 2025 validation, "
                    f"**{res['oof_observed_weighted_pct']:.1f}%** of children in this tier had elevated blood lead (≥ 5 µg/dL)."
                )
            elif tier == "Moderate":
                st.info(
                    f"**Moderate Risk Tier ({res['probability_pct']:.1f}%):** Close to the national weighted prevalence "
                    f"({res['national_weighted_avg_pct']:.1f}%). Review household risk drivers below."
                )
            else:
                st.success(
                    f"**Low Risk Tier ({res['probability_pct']:.1f}%):** Below the national weighted prevalence "
                    f"({res['national_weighted_avg_pct']:.1f}%), though routine lead-safe practices remain worthwhile."
                )

            render_contribution_chart(res["contributions"])

            with st.expander("View Exact Factor Contribution Table"):
                st.dataframe(res["contributions"], use_container_width=True, hide_index=True)

            st.subheader("Recommended Next Steps")
            for rec in res["recommendations"]:
                st.markdown(f"**[{rec['priority']}] {rec['title']}**  \n{rec['detail']}")

    # =========================================================================
    # TAB 2: MODEL EVIDENCE & RESEARCH FINDINGS
    # =========================================================================
    with tab_evidence:
        st.subheader("1. Production Model Validation Summary (5-Fold PSU-Grouped CV)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Weighted ROC-AUC", f"{metrics['roc_auc_weighted']:.4f}")
        c2.metric("Unweighted ROC-AUC", f"{metrics['roc_auc_unweighted']:.4f} (±{metrics['roc_auc_fold_sd']:.4f})")
        c3.metric("Weighted PR-AUC", f"{metrics['pr_auc_weighted']:.4f}")
        c4.metric("Weighted ECE / Brier", f"{metrics['ece_weighted']:.4f} / {metrics['brier_weighted']:.4f}")

        st.subheader("2. Out-of-Fold 4-Tier Risk Stratification")
        tier_df = pd.DataFrame(bundle["risk_tiers"]["oof_validation"])
        tier_df.columns = [
            "Risk Tier",
            "Validated Children (n)",
            "Weighted Pop Share (%)",
            "Observed Unweighted Rate (%)",
            "Observed Weighted Rate (%)",
        ]
        st.dataframe(tier_df, use_container_width=True, hide_index=True)

        if BENCHMARK_CSV.exists():
            st.subheader("3. Multi-Model & Feature-Tier Benchmark (Geo vs. Parent vs. Parent + Soil)")
            bench = pd.read_csv(BENCHMARK_CSV)
            st.dataframe(bench, use_container_width=True, hide_index=True)
            st.markdown(
                "- **Research Question 1 (Parent Accessibility):** Regularized Logistic Regression outperformed Random Forest, "
                "HistGradientBoosting, and MLP Neural Net across all 6 tiers, adding **+0.0123 weighted ROC-AUC** and "
                "**+0.0997 within-district AUC** over geography alone.\n"
                "- **Research Question 2 (Lab Soil Value):** Adding 4 measured soil heavy metals (`S_Lead`, `S_As`, `S_Hg`, `S_Cd`) "
                "produced only **+0.0017 weighted ROC-AUC** lift on the best model (`0.7550 → 0.7567`), proving parent-accessible "
                "screening captures nearly all predictive value available without laboratory soil assays."
            )

        if SUMMARY_JSON.exists():
            st.subheader("4. Verified Non-Geographic Odds Ratios (Survey-Weighted Logistic Regression)")
            summary_data = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
            or_df = pd.DataFrame(summary_data["non_geographic_odds_ratios"])
            st.dataframe(or_df, use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 3: METHODOLOGY & SCOPE LIMITS
    # =========================================================================
    with tab_method:
        st.subheader("Cohort Construction & Data Integrity")
        st.markdown(
            "- **Source:** Bangladesh MICS 2025 (MICS7) `CH.sav` (`24,680` children) linked many-to-one with `HH.sav` (`62,980` households) on `(HH1, HH2)` with 100% match rate.\n"
            "- **Validated Blood-Lead Cohort:** `10,622` children aged **12–59 months** after excluding `12` invalid `C_Lead == 999` lab records (`0` infants under 12 months were tested).\n"
            "- **Targets:** Primary binary target `Elevated_Lead` at **≥ 5 µg/dL** (`31.70%` unweighted / `38.51%` survey-weighted using `chblweight`); secondary sensitivity benchmark at **≥ 3.5 µg/dL**.\n"
            "- **Leakage Prevention:** Cluster-grouped 5-fold CV (`StratifiedGroupKFold` on `PSU`) prevents neighboring children from leaking spatial context across folds."
        )

        st.subheader("Confounding Traps Identified & Excluded")
        st.markdown(
            "- **Piped Natural Gas (`EU1 == 3`):** Showed `73.6%` raw elevated lead, but collapsed to `-1.6pp to +3.0pp` inside Dhaka/Narayanganj/Chattogram—purely an industrial-corridor proxy. Excluded to prevent false advice.\n"
            "- **Water Filtration (`HC7G`) & Water Treatment (`WS9`):** Spiked in raw data (`52.6%` Wtd) because urban metro residents treat municipal water. Excluded.\n"
            "- **PCA Wealth Index (`windex5`):** Inverted positively (`30.2% → 51.9%` Wtd) due to urban concentration and unanswerable by ordinary parents. Replaced with **Mother's Education (`melevel`)** and **Sleeping Rooms (`HC3`)**."
        )

        st.subheader("Scientific Limitations (Why LeadSight Is Non-Diagnostic)")
        st.markdown(
            "- **Unmeasured Micro-Pathways:** MICS7 does not measure household turmeric/spice adulteration, aluminum/brass cookware leaching, or exact meter-level distance to informal smelters—limiting within-district discrimination (`0.5475` mean within-district AUC).\n"
            "- **Eligibility Boundary:** Strictly validated for children aged **12–59 months** in Bangladesh. Do not extrapolate to infants under 12 months or older children.\n"
            "- **Decision Boundary:** A low estimate cannot guarantee a blood lead level below 5 µg/dL, and a high estimate requires confirmation via a clinical venous blood lead test."
        )


if __name__ == "__main__":
    main()

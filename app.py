"""Industrial Predictive Maintenance & Equipment Failure Risk Engine.

Interactive Streamlit Operations Cockpit.
"""

from __future__ import annotations

import logging
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from src.explainability import diagnose_telemetry_reading

# Configure page
st.set_page_config(
    page_title="Predictive Maintenance Risk Engine",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Preset Industrial Scenarios
PRESETS = {
    "Select a Scenario...": None,
    "🟢 Nominal Healthy Machining": {
        "product_type": "M",
        "air_temperature_k": 298.5,
        "process_temperature_k": 308.7,
        "rotational_speed_rpm": 1550.0,
        "torque_nm": 40.0,
        "tool_wear_min": 35.0,
    },
    "🟡 Worn Cutter Degradation (Approaching Limit)": {
        "product_type": "L",
        "air_temperature_k": 300.2,
        "process_temperature_k": 310.4,
        "rotational_speed_rpm": 1480.0,
        "torque_nm": 48.0,
        "tool_wear_min": 185.0,
    },
    "🔴 Heat Dissipation Failure (Coolant Loss)": {
        "product_type": "L",
        "air_temperature_k": 302.5,
        "process_temperature_k": 309.8,
        "rotational_speed_rpm": 1310.0,
        "torque_nm": 56.0,
        "tool_wear_min": 120.0,
    },
    "🔴 Overstrain Mechanical Failure (High Torque + Wear)": {
        "product_type": "L",
        "air_temperature_k": 301.0,
        "process_temperature_k": 310.5,
        "rotational_speed_rpm": 1380.0,
        "torque_nm": 68.0,
        "tool_wear_min": 215.0,
    },
    "🔴 Power Failure (Motor Overload)": {
        "product_type": "M",
        "air_temperature_k": 299.0,
        "process_temperature_k": 309.0,
        "rotational_speed_rpm": 2750.0,
        "torque_nm": 65.0,
        "tool_wear_min": 90.0,
    },
}

# App Header
st.markdown(
    """
    <div style="background: linear-gradient(90deg, #1E3A8A 0%, #0F172A 100%); padding: 22px 28px; border-radius: 12px; margin-bottom: 24px; color: white;">
        <h1 style="margin: 0; font-size: 28px; font-weight: 700;">⚙️ Predictive Maintenance & Equipment Failure Risk Engine</h1>
        <p style="margin: 6px 0 0 0; font-size: 14px; opacity: 0.85;">Industrial IoT Telemetry Intelligence • Cost-Sensitive Gradient Boosting • Real-Time TreeSHAP Diagnostics</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Navigation
tabs = st.tabs(
    [
        "🎛️ Real-Time Telemetry Risk Simulator",
        "⚡ IoT Stream Replay",
        "📈 Model Benchmarks & Cost Matrix",
        "🔬 Statistical Screening (ANOVA & χ²)",
    ]
)

# --- TAB 1: REAL-TIME SIMULATOR ---
with tabs[0]:
    col_input, col_diag = st.columns([1.1, 1.4], gap="large")

    with col_input:
        st.subheader("1. Machine Telemetry Inputs")

        scenario = st.selectbox(
            "Load Quick Preset Scenario:",
            list(PRESETS.keys()),
            index=0,
        )

        preset_vals = PRESETS.get(scenario) or {
            "product_type": "L",
            "air_temperature_k": 300.0,
            "process_temperature_k": 310.0,
            "rotational_speed_rpm": 1500.0,
            "torque_nm": 45.0,
            "tool_wear_min": 110.0,
        }

        col_a, col_b = st.columns(2)
        with col_a:
            p_type = st.selectbox(
                "Product Quality Variant",
                ["L", "M", "H"],
                index=["L", "M", "H"].index(preset_vals["product_type"]),
                help="L: Low variant (50%), M: Medium (30%), H: High variant (20%)",
            )
            air_temp = st.slider(
                "Air Temperature [K]",
                295.0,
                305.0,
                float(preset_vals["air_temperature_k"]),
                step=0.1,
                help="Ambient shop floor temperature",
            )
            proc_temp = st.slider(
                "Process Temperature [K]",
                305.0,
                315.0,
                float(preset_vals["process_temperature_k"]),
                step=0.1,
                help="Tool-workpiece interface temperature",
            )

        with col_b:
            speed_rpm = st.slider(
                "Rotational Speed [RPM]",
                1100.0,
                2900.0,
                float(preset_vals["rotational_speed_rpm"]),
                step=10.0,
                help="Spindle motor rotation speed",
            )
            torque = st.slider(
                "Torque [Nm]",
                10.0,
                80.0,
                float(preset_vals["torque_nm"]),
                step=0.5,
                help="Mechanical cutting torque",
            )
            tool_wear = st.slider(
                "Tool Wear [min]",
                0.0,
                250.0,
                float(preset_vals["tool_wear_min"]),
                step=1.0,
                help="Cumulative machining minutes on cutter",
            )

        # Computed Real-Time Physics Signals
        mech_power = torque * (speed_rpm * 2 * np.pi / 60.0)
        temp_delta = proc_temp - air_temp
        overstrain = tool_wear * torque

        st.markdown("#### Physics-Engineered Indicators")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Mechanical Power", f"{mech_power:,.0f} W", delta=f"{mech_power - 6200:,.0f}W vs norm")
        col_m2.metric("Thermal Gradient (ΔT)", f"{temp_delta:.1f} K", delta=f"{temp_delta - 10.0:.1f}K vs norm")
        col_m3.metric("Overstrain Index", f"{overstrain:,.0f}", delta=f"{overstrain - 4200:,.0f} vs norm")

    with col_diag:
        st.subheader("2. Real-Time Failure Risk & SHAP Diagnostics")

        # Telemetry Dictionary
        telemetry_input = {
            "udi": 1,
            "product_id": f"{p_type}99999",
            "product_type": p_type,
            "air_temperature_k": air_temp,
            "process_temperature_k": proc_temp,
            "rotational_speed_rpm": speed_rpm,
            "torque_nm": torque,
            "tool_wear_min": tool_wear,
            "machine_failure": 0,
            "twf": 0,
            "hdf": 0,
            "pwf": 0,
            "osf": 0,
            "rnf": 0,
        }

        with st.spinner("Analyzing telemetry stream & calculating SHAP attributions..."):
            diag_result = diagnose_telemetry_reading(telemetry_input)

        prob = diag_result["failure_probability"]
        tier = diag_result["risk_tier"]

        # Risk Banner
        if tier == "GREEN":
            st.success(f"### 🟢 Risk Level: {prob*100:.1f}% — {diag_result['status']}")
        elif tier == "AMBER":
            st.warning(f"### 🟡 Risk Level: {prob*100:.1f}% — {diag_result['status']}")
        else:
            st.error(f"### 🔴 Risk Level: {prob*100:.1f}% — {diag_result['status']}")

        st.progress(min(prob, 1.0))

        # Prescription Box
        st.info(f"**Automated Maintenance Prescription:**\n{diag_result['prescription']}")

        # Root Cause Attribution
        st.markdown("#### Root Cause Decomposition (TreeSHAP)")
        if diag_result["top_drivers"]:
            driver_df = pd.DataFrame(diag_result["top_drivers"])
            driver_df = driver_df.rename(
                columns={
                    "sensor": "Sensor Signal",
                    "shap_attribution": "Risk Push (+SHAP)",
                    "normalized_value": "Z-Score",
                }
            )
            st.dataframe(driver_df, use_container_width=True, hide_index=True)

            # Horizontal Bar Chart
            fig, ax = plt.subplots(figsize=(6, 2.5))
            y_pos = np.arange(len(driver_df))
            ax.barh(y_pos, driver_df["Risk Push (+SHAP)"], color="#EF4444", align="center")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(driver_df["Sensor Signal"], fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel("Attribution Contribution to Failure Risk", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.write("All sensors operating well within normal baseline distributions.")


# --- TAB 2: IOT STREAM REPLAY ---
with tabs[1]:
    st.subheader("⚡ Live IoT Telemetry Stream Replay")
    st.caption("Simulates real-time telemetry streaming from CNC milling machines on the factory floor.")

    test_file = PROCESSED_DATA_DIR / "test_raw.csv"
    if test_file.exists():
        test_df = pd.read_csv(test_file)
        sample_size = st.slider("Select Stream Window Size (Cycles)", 10, 100, 25)

        filter_option = st.radio(
            "Filter Stream By:",
            ["All Operations", "Only Actual Breakdown Events", "Only Healthy Operations"],
            horizontal=True,
        )

        if filter_option == "Only Actual Breakdown Events":
            stream_batch = test_df[test_df["machine_failure"] == 1].head(sample_size)
        elif filter_option == "Only Healthy Operations":
            stream_batch = test_df[test_df["machine_failure"] == 0].head(sample_size)
        else:
            stream_batch = test_df.head(sample_size)

        st.dataframe(
            stream_batch[
                [
                    "udi",
                    "product_type",
                    "air_temperature_k",
                    "process_temperature_k",
                    "rotational_speed_rpm",
                    "torque_nm",
                    "tool_wear_min",
                    "machine_failure",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        if st.button("🚀 Run Batch Stream Diagnostic", type="primary"):
            st.write(f"Analyzing {len(stream_batch)} streaming cycles...")
            results_list = []
            for _, row in stream_batch.iterrows():
                row_dict = row.to_dict()
                res = diagnose_telemetry_reading(row_dict)
                results_list.append(
                    {
                        "Cycle UDI": int(row["udi"]),
                        "Product": row["product_type"],
                        "Actual Failure": "🔴 FAIL" if row["machine_failure"] == 1 else "🟢 Normal",
                        "Predicted Risk": f"{res['failure_probability']*100:.1f}%",
                        "Risk Tier": res["risk_tier"],
                        "Prescription": res["prescription"],
                    }
                )
            st.dataframe(pd.DataFrame(results_list), use_container_width=True, hide_index=True)


# --- TAB 3: BENCHMARKS & COST MATRIX ---
with tabs[2]:
    st.subheader("📈 Model Benchmarks & Industrial Cost Matrix Optimization")

    col_bench, col_charts = st.columns([1, 1], gap="medium")
    with col_bench:
        benchmark_file = REPORTS_DIR / "model_benchmarks_report.md"
        if benchmark_file.exists():
            with open(benchmark_file, "r", encoding="utf-8") as f:
                st.markdown(f.read())

    with col_charts:
        pr_chart = FIGURES_DIR / "precision_recall_curves.png"
        cost_chart = FIGURES_DIR / "cost_vs_threshold.png"
        cm_chart = FIGURES_DIR / "confusion_matrix.png"

        if pr_chart.exists():
            st.image(str(pr_chart), caption="Precision-Recall Curves on Test Set")
        if cost_chart.exists():
            st.image(str(cost_chart), caption="Industrial Cost vs. Classification Threshold Curve")
        if cm_chart.exists():
            st.image(str(cm_chart), caption="Confusion Matrix at Cost-Optimal Threshold")


# --- TAB 4: STATISTICAL SCREENING ---
with tabs[3]:
    st.subheader("🔬 Statistical Hypothesis Testing Gate (ANOVA & χ²)")
    stats_file = REPORTS_DIR / "statistical_tests_report.md"
    if stats_file.exists():
        with open(stats_file, "r", encoding="utf-8") as f:
            st.markdown(f.read())

    shap_chart = FIGURES_DIR / "shap_summary.png"
    if shap_chart.exists():
        st.markdown("### Fleet-Wide Root Cause Ranking (TreeSHAP)")
        st.image(str(shap_chart), caption="TreeSHAP Feature Attribution Summary")

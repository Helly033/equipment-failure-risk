"""Statistical Hypothesis Testing and Sensor Screening Gate.

Executes One-Way ANOVA F-Tests and Chi-Square Contingency Tests
to mathematically validate which sensor signals and physics features
are statistically significant predictors of failure.
"""

from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"


def run_anova_feature_screening(
    df: pd.DataFrame, target_col: str = "machine_failure", alpha: float = 0.05
) -> pd.DataFrame:
    """Perform One-Way ANOVA F-Test on all continuous sensor features."""
    results = []
    healthy = df[df[target_col] == 0]
    failing = df[df[target_col] == 1]

    # Candidate numeric features (exclude IDs and failure labels)
    exclude_cols = {
        "udi",
        "product_id",
        "product_type",
        "machine_failure",
        "twf",
        "hdf",
        "pwf",
        "osf",
        "rnf",
    }
    numeric_features = [
        c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols
    ]

    for feat in numeric_features:
        group_h = healthy[feat].dropna()
        group_f = failing[feat].dropna()

        # One-way ANOVA
        f_stat, p_val = stats.f_oneway(group_h, group_f)

        # Effect Size: Eta-Squared (SS_between / SS_total)
        overall_mean = df[feat].mean()
        ss_between = len(group_h) * (group_h.mean() - overall_mean) ** 2 + len(
            group_f
        ) * (group_f.mean() - overall_mean) ** 2
        ss_total = ((df[feat] - overall_mean) ** 2).sum()
        eta_squared = ss_between / ss_total if ss_total > 0 else 0.0

        results.append(
            {
                "feature": feat,
                "f_statistic": round(f_stat, 4),
                "p_value": p_val,
                "p_value_formatted": f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}",
                "eta_squared": round(eta_squared, 4),
                "mean_healthy": round(group_h.mean(), 3),
                "mean_failing": round(group_f.mean(), 3),
                "significant": p_val < alpha,
            }
        )

    results_df = (
        pd.DataFrame(results)
        .sort_values(by="f_statistic", ascending=False)
        .reset_index(drop=True)
    )
    return results_df


def run_chi_square_product_type(
    df: pd.DataFrame, target_col: str = "machine_failure", alpha: float = 0.05
) -> dict:
    """Perform Chi-Square Contingency Test of Independence between Product Type and Failure."""
    contingency_table = pd.crosstab(df["product_type"], df[target_col])
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)

    # Cramér's V (Effect size for contingency table)
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0.0

    return {
        "contingency_table": contingency_table,
        "chi2_statistic": round(chi2, 4),
        "p_value": p_val,
        "p_value_formatted": f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}",
        "degrees_of_freedom": dof,
        "cramers_v": round(cramers_v, 4),
        "significant": p_val < alpha,
    }


def generate_statistical_report(
    anova_df: pd.DataFrame, chi2_results: dict, output_path: Path
) -> None:
    """Export formatted markdown report summarizing the statistical screening results."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "# Statistical Hypothesis Testing & Feature Screening Report",
        "",
        "## 1. Continuous Features: One-Way ANOVA F-Tests",
        "",
        "Tests null hypothesis $H_0$: Mean sensor reading is identical across normal vs. breakdown states ($p < 0.05$ rejects $H_0$).",
        "",
        "| Rank | Feature | F-Statistic | p-value | $\\eta^2$ Effect Size | Mean (Normal) | Mean (Failing) | Decision |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for idx, row in anova_df.iterrows():
        decision = "✅ Retained (Significant)" if row["significant"] else "❌ Rejected (Noise)"
        report_lines.append(
            f"| {idx + 1} | `{row['feature']}` | {row['f_statistic']:,} | {row['p_value_formatted']} | "
            f"{row['eta_squared']} | {row['mean_healthy']} | {row['mean_failing']} | {decision} |"
        )

    report_lines.extend(
        [
            "",
            "---",
            "",
            "## 2. Categorical Metadata: Chi-Square ($\\chi^2$) Contingency Test",
            "",
            "Tests null hypothesis $H_0$: Product Quality Variant (`L`, `M`, `H`) is independent of Machine Failure.",
            "",
            f"- **$\\chi^2$ Statistic:** {chi2_results['chi2_statistic']}",
            f"- **p-value:** {chi2_results['p_value_formatted']}",
            f"- **Degrees of Freedom:** {chi2_results['degrees_of_freedom']}",
            f"- **Cramér's V (Effect Size):** {chi2_results['cramers_v']}",
            f"- **Conclusion:** {'Statistically Significant correlation (Reject H0)' if chi2_results['significant'] else 'Independent (Fail to reject H0)'}",
            "",
            "### Contingency Matrix (Observed Frequencies)",
            "",
            chi2_results["contingency_table"].to_markdown(),
            "",
        ]
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Statistical screening report exported to: {output_path}")


def run_statistical_screening() -> tuple[pd.DataFrame, dict]:
    """Execute complete statistical screening pipeline on train dataset."""
    train_eng_path = PROCESSED_DATA_DIR / "train_engineered.csv"
    if not train_eng_path.exists():
        raise FileNotFoundError(
            "Engineered train dataset missing! Run src.features first."
        )

    df_train = pd.read_csv(train_eng_path)
    logger.info(f"Loaded {len(df_train):,} training records for statistical screening.")

    # 1. Run ANOVA F-Tests
    anova_results = run_anova_feature_screening(df_train)
    logger.info(f"ANOVA screening completed on {len(anova_results)} features.")

    # 2. Run Chi-Square Test
    chi2_results = run_chi_square_product_type(df_train)
    logger.info(
        f"Chi-Square test on Product Type: chi2={chi2_results['chi2_statistic']}, "
        f"p-val={chi2_results['p_value_formatted']}"
    )

    # 3. Export Report
    report_path = REPORTS_DIR / "statistical_tests_report.md"
    generate_statistical_report(anova_results, chi2_results, report_path)

    # Print top 5 strongest predictors
    logger.info("Top 5 Strongest Statistical Predictors of Breakdown:")
    for _, row in anova_results.head(5).iterrows():
        logger.info(
            f"  - {row['feature']:<25} F-Stat: {row['f_statistic']:>10.2f} (p-val: {row['p_value_formatted']})"
        )

    return anova_results, chi2_results


if __name__ == "__main__":
    logger.info("--- Starting Statistical Hypothesis Screening Gate ---")
    run_statistical_screening()
    logger.info("--- Statistical Screening Gate Completed Successfully ---")

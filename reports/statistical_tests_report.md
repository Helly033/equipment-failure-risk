# Statistical Hypothesis Testing & Feature Screening Report

## 1. Continuous Features: One-Way ANOVA F-Tests

Tests null hypothesis $H_0$: Mean sensor reading is identical across normal vs. breakdown states ($p < 0.05$ rejects $H_0$).

| Rank | Feature | F-Statistic | p-value | $\eta^2$ Effect Size | Mean (Normal) | Mean (Failing) | Decision |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `power_to_thermal_ratio` | 424.7397 | 5.4032e-92 | 0.0504 | 629.237 | 786.205 | ✅ Retained (Significant) |
| 2 | `torque_nm` | 299.6527 | 6.1763e-66 | 0.0361 | 39.647 | 50.171 | ✅ Retained (Significant) |
| 3 | `overstrain_index` | 285.8221 | 4.9864e-63 | 0.0345 | 4204.599 | 7100.111 | ✅ Retained (Significant) |
| 4 | `mechanical_power_w` | 255.6572 | 1.1397e-56 | 0.031 | 6247.278 | 7290.524 | ✅ Retained (Significant) |
| 5 | `torque_rolling_std` | 177.3586 | 4.8723e-40 | 0.0217 | 9.309 | 12.153 | ✅ Retained (Significant) |
| 6 | `torque_rate_of_change` | 165.0414 | 2.0963e-37 | 0.0202 | -0.379 | 10.807 | ✅ Retained (Significant) |
| 7 | `temp_difference_k` | 100.3442 | 1.7610e-23 | 0.0124 | 10.021 | 9.406 | ✅ Retained (Significant) |
| 8 | `tool_wear_min` | 79.5102 | 5.8668e-19 | 0.0098 | 106.503 | 141.387 | ✅ Retained (Significant) |
| 9 | `speed_rolling_std` | 79.3413 | 6.3850e-19 | 0.0098 | 154.313 | 204.795 | ✅ Retained (Significant) |
| 10 | `air_temperature_k` | 59.5953 | 1.3060e-14 | 0.0074 | 299.973 | 300.923 | ✅ Retained (Significant) |
| 11 | `temp_diff_rolling_mean` | 13.9504 | 0.0002 | 0.0017 | 10.004 | 9.901 | ✅ Retained (Significant) |
| 12 | `speed_rate_of_change` | 13.6223 | 0.0002 | 0.0017 | 1.981 | -56.465 | ✅ Retained (Significant) |
| 13 | `process_temperature_k` | 13.3681 | 0.0003 | 0.0017 | 309.995 | 310.329 | ✅ Retained (Significant) |
| 14 | `rotational_speed_rpm` | 12.7957 | 0.0003 | 0.0016 | 1540.711 | 1500.731 | ✅ Retained (Significant) |
| 15 | `thermal_dissipation_ratio` | 0.8874 | 0.3462 | 0.0001 | 0.007 | 0.007 | ❌ Rejected (Noise) |
| 16 | `torque_wear_ratio` | 0.8113 | 0.3678 | 0.0001 | 1.306 | 1.567 | ❌ Rejected (Noise) |

---

## 2. Categorical Metadata: Chi-Square ($\chi^2$) Contingency Test

Tests null hypothesis $H_0$: Product Quality Variant (`L`, `M`, `H`) is independent of Machine Failure.

- **$\chi^2$ Statistic:** 18.0942
- **p-value:** 0.0001
- **Degrees of Freedom:** 2
- **Cramér's V (Effect Size):** 0.0476
- **Conclusion:** Statistically Significant correlation (Reject H0)

### Contingency Matrix (Observed Frequencies)

| product_type   |    0 |   1 |
|:---------------|-----:|----:|
| H              |  773 |  16 |
| L              | 4633 | 197 |
| M              | 2323 |  58 |

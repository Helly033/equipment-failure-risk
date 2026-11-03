# Industrial Model Benchmark Evaluation & Cost Analysis

Evaluated on 2,000 holdout machine cycles (68 actual breakdown events).

### Cost Assumptions:
- **False Negative (Missed Breakdown):** $10,000.00
- **False Positive (Unnecessary Check):** $200.00
- **True Positive (Scheduled Replacement):** $1,500.00

### Performance Matrix

| Model                        |   Precision |   Recall |   F2-Score |   PR-AUC |   Missed Failures (FN) | Total Plant Cost   |
|:-----------------------------|------------:|---------:|-----------:|---------:|-----------------------:|:-------------------|
| Naive Baseline (All 0s)      |       0     |    0     |      0     |   0.034  |                     68 | $680,000.00        |
| Logistic Regression (τ=0.50) |       0.189 |    0.853 |      0.501 |   0.4647 |                     10 | $236,800.00        |
| Random Forest (τ=0.50)       |       0.961 |    0.721 |      0.759 |   0.8757 |                     19 | $263,900.00        |
| XGBoost (τ=0.20)             |       0.558 |    0.926 |      0.818 |   0.8924 |                      5 | $154,500.00        |
| LightGBM (τ=0.50)            |       0.789 |    0.824 |      0.816 |   0.8834 |                     12 | $207,000.00        |

### Key Finding:
By using cost-sensitive XGBoost with optimal threshold $\tau^* = 0.20$, the engine captures **92.6%** of breakdowns and saves **$525,500.00** compared to unmonitored baseline operation.
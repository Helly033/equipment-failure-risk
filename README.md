# Predictive Maintenance & Equipment Failure Risk Engine

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-EB2529?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-009688?style=for-the-badge)](https://shap.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> An end-to-end Industrial IoT Machine Learning system that predicts manufacturing equipment breakdowns before catastrophic failure, quantifies downtime financial risk, and explains root causes using Explainable AI (SHAP).

---

## 📌 Executive Summary

Unplanned industrial downtime costs manufacturing plants an estimated **$50 billion annually**. Traditional maintenance relies on either:
1. **Reactive Maintenance:** Run-to-failure (causes costly emergency shutdowns).
2. **Preventative Maintenance:** Fixed-interval replacements (wastes thousands of hours of remaining part life).

This project implements an **Industrial Risk Engine** that ingests real-time telemetry (temperatures, rotational speeds, torque, and tool wear), applies **physics-informed feature engineering**, screens signals using **statistical hypothesis testing (ANOVA & $\chi^2$)**, and predicts breakdowns using **cost-optimized gradient boosting**.

---

## 🏗️ Enterprise System Architecture

```mermaid
flowchart TD
    %% TIER 1: CLIENTS
    subgraph CLIENTS["1. Clients & Plant Consumers"]
        direction LR
        C_DASH["Operations Dashboard<br/>(Streamlit / Web UI)"]
        C_HMI["Plant Floor HMI Terminal<br/>(Operator Console / Tablet)"]
        C_ALERT["Emergency Alert / Pager<br/>(SCADA Interlock / Webhook)"]
    end

    %% TIER 2: CONTROL PLANE
    subgraph CONTROL_PLANE["2. Control Plane — apps/core (FastAPI + ML Services)"]
        direction TB
        subgraph CP_GW["API & Ingestion Gateway"]
            API_GW["REST API /api/v1 & WebSockets"]
            AUTH["UnifiedAuthGuard<br/>(JWT / API Key / HMAC)"]
            INGEST_SVC["Telemetry Ingestion Service<br/>(Schema & Range Validator)"]
            API_GW --> AUTH --> INGEST_SVC
        end

        subgraph CP_ENG["Analytics & ML Engine"]
            FE_ENGINE["Physics & Signal Engine<br/>- Mechanical Power (W)<br/>- Thermal Gradient ΔT<br/>- Overstrain Factor<br/>- Rolling Volatility (σ)"]
            STATS_GATE["Statistical Validation Gate<br/>- ANOVA F-Test Screening<br/>- Chi-Square (χ²) Independence<br/>- Sensor Drift Detector"]
            INF_WORKER["ML Inference Worker<br/>(XGBoost / LightGBM)"]
            COST_POL["Cost Policy Optimizer<br/>(Minimizes $10k FN Downtime Cost)"]
            SHAP_WORKER["SHAP Diagnostic Worker<br/>(Local Waterfall Root Cause)"]
            ALERT_SVC["Alert & Dispatch Service<br/>(Risk Tiers: Green / Amber / Red)"]

            INGEST_SVC --> FE_ENGINE
            FE_ENGINE --> STATS_GATE
            STATS_GATE --> INF_WORKER
            INF_WORKER --> COST_POL
            COST_POL --> SHAP_WORKER
            SHAP_WORKER --> ALERT_SVC
        end
    end

    %% TIER 3: DATA PLANE
    subgraph DATA_PLANE["3. Data Plane — apps/edge (Streaming & Telemetry Pipeline)"]
        direction TB
        subgraph EDGE_MACHINES["Industrial Machine Fleet (Edge Sensors)"]
            M_CNC1["CNC Milling Cell #1<br/>(Temp, RPM, Torque, Wear)"]
            M_CNC2["CNC Milling Cell #2<br/>(Temp, RPM, Torque, Wear)"]
            M_CNC3["CNC Milling Cell #N<br/>(Temp, RPM, Torque, Wear)"]
        end

        subgraph EDGE_BROKER["Edge Ingestion & Buffering"]
            EDGE_GW["Edge IoT Gateway<br/>(OPC-UA / MQTT Broker)"]
            STREAM_WORKER["Stream Consumer Worker<br/>(Window Aggregator)"]
            REDIS_BUFFER[("Redis Ring Buffer<br/>(Rolling 30-Cycle Window)")]
        end

        M_CNC1 --> EDGE_GW
        M_CNC2 --> EDGE_GW
        M_CNC3 --> EDGE_GW
        EDGE_GW --> STREAM_WORKER
        STREAM_WORKER <--> REDIS_BUFFER
    end

    %% TIER 4: STATEFUL INFRASTRUCTURE
    subgraph INFRA["4. Stateful Infrastructure"]
        direction LR
        DB_REDIS[("Redis 7<br/>(Sliding Buffer & Cache)")]
        DB_TS[("InfluxDB / TimescaleDB<br/>(Telemetry Historian)")]
        DB_POSTGRES[("PostgreSQL 16<br/>(Machines, Logs, Work Orders)")]
        DB_S3[("MinIO / S3<br/>(Model Registry & Artifacts)")]
        QUEUE_KAFKA[("Kafka / Redpanda<br/>(Event Stream Broker)")]
    end

    %% INTER-TIER FLOWS
    C_DASH -- "1. Telemetry Query & What-If Simulation" --> API_GW
    API_GW -- "2. Risk Probabilities & SHAP Waterfall" --> C_DASH
    ALERT_SVC -- "3. Emergency Throttle / Pager Alarm" --> C_ALERT
    ALERT_SVC -- "4. Operator Prescription Ticket" --> C_HMI

    STREAM_WORKER -- "5. Sensor Stream Batch" --> INGEST_SVC
    STREAM_WORKER -- "6. High-Frequency Pub/Sub" --> QUEUE_KAFKA
    QUEUE_KAFKA --> INGEST_SVC

    INF_WORKER -- "7. Load Model & Encoders" --> DB_S3
    ALERT_SVC -- "8. Persist Prediction & Incident" --> DB_POSTGRES
    STREAM_WORKER -- "9. Store Raw Sensor Telemetry" --> DB_TS
    REDIS_BUFFER -. "10. Cache Sync" .-> DB_REDIS
    C_HMI -. "11. Direct Machine Control" .-> EDGE_GW
```

*For complete architectural specifications, see [docs/architecture.md](docs/architecture.md).*

---

## 🛠️ Technology Stack

| Domain | Technology / Library | Role in Project |
| :--- | :--- | :--- |
| **Language** | `Python 3.10 / 3.11` | Primary development environment |
| **Data Processing** | `Pandas`, `NumPy` | Vectorized sensor transformations and rolling statistics |
| **Statistical Analysis** | `SciPy (scipy.stats)` | ANOVA F-tests and Chi-Square tests of independence |
| **Machine Learning** | `Scikit-Learn` | Preprocessing pipelines, baselines, and cross-validation |
| **Gradient Boosting** | `XGBoost`, `LightGBM` | High-precision classification on imbalanced tabular data |
| **Model Interpretability** | `SHAP` | TreeSHAP global impact and local waterfall explanations |
| **Visualization** | `Matplotlib`, `Seaborn` | EDA charts, PR curves, and confusion matrices |
| **Application / UI** | `Streamlit` | Interactive operational risk dashboard |

---

## 🔬 Core Innovations & Workflow

### 1. Physics-Informed Feature Engineering
Rather than treating sensors as disconnected numbers, we introduce thermodynamic and mechanical equations:
* **Mechanical Power ($W$):** $P = \text{Torque} \times \left(\text{RPM} \times \frac{2\pi}{60}\right)$
* **Thermal Gradient ($\Delta T$):** $\Delta T = \text{Process Temp} - \text{Air Temp}$
* **Overstrain Index:** $\text{Tool Wear} \times \text{Torque}$

### 2. Statistical Screening Gate
* **One-Way ANOVA:** Validates that telemetry distributions across failure vs. healthy states are statistically distinct ($p < 0.05$).
* **Chi-Square ($\chi^2$):** Analyzes whether product variant grades (`L`, `M`, `H`) correlate with specific failure modes (Tool Wear, Heat Dissipation, Overstrain).

### 3. Business Cost Matrix & Threshold Tuning
In predictive maintenance, **False Negatives** (missed failures) are exponentially more expensive than **False Positives** (unnecessary 15-minute checkups).
* Cost of False Negative: **$10,000** (unplanned downtime, broken spindle)
* Cost of False Positive: **$200** (technician inspection)
* We optimize the classification threshold $\tau$ to minimize **Total Expected Industrial Cost**, not arbitrary accuracy.

### 4. Explainable AI with SHAP
Every alert provides maintenance crews with an automated diagnostic:
> *"Machine #1042 Risk: 84% [CRITICAL]. Primary drivers: Tool Wear reached 215 min (+42% risk), Mechanical Power dropped below safe threshold (+26% risk)."*

---

## 📂 Repository Layout

```
equipment-failure-risk/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/                       # Immutable raw sensor telemetry
│   └── processed/                 # Engineered features & split datasets
│
├── docs/
│   └── architecture.md            # Detailed system architecture document
│
├── notebooks/
│   ├── 01_eda_and_statistical_tests.ipynb  # ANOVA, Chi2, sensor distributions
│   └── 02_model_experimentation.ipynb      # Baseline vs XGBoost & Cost Curve
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Data fetching & train/test split
│   ├── features.py                # Physics equations & rolling statistics
│   ├── stats_testing.py           # Automated ANOVA & Chi-Square screening
│   ├── models.py                  # Training pipeline (XGBoost, Random Forest)
│   ├── evaluation.py              # Cost matrix, PR-AUC, threshold optimization
│   └── explainability.py          # SHAP calculation & plot generators
│
├── models/                        # Serialized champion models and scalers
├── reports/
│   └── figures/                   # Exported SHAP plots and evaluation charts
│
└── app.py                         # Interactive Streamlit Risk Engine application
```

---

## 🚀 Quickstart

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/<your-username>/equipment-failure-risk.git
cd equipment-failure-risk

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Statistical Analysis & Model Pipeline
```bash
python -m src.data_loader
python -m src.stats_testing
python -m src.models
```

### 3. Launch the Interactive Dashboard
```bash
streamlit run app.py
```

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

# Zomato Business Intelligence & Delivery Intelligence Platform

This repository is a complete, runnable end-to-end BI + machine-learning project built specifically for the supplied Zomato-style ZIP dataset. It addresses the source-data defects found during analysis instead of silently training on them.

## 1. What the project contains

The workflow is:

**Raw CSVs → data-quality audit → cleaning → validation → feature engineering → PostgreSQL/SQL analytics → EDA → delivery-time regression → 60-day churn classification → Streamlit dashboard → FastAPI scoring API → Power BI measures**

The project includes:

- all supplied raw CSV files in `data/raw/`
- the original PRD
- the original schema/query SQL for reference
- a corrected constrained PostgreSQL schema
- corrected business queries and Power BI views
- robust mixed-date parsing with ambiguity flags
- city/categorical normalization
- duplicate and orphan handling
- invalid numeric/rating cleanup
- line-item based reconstruction of order food cost
- order-level financial reconciliation
- time-aware weather/traffic feature joins
- leakage-safe prior-history features
- time-based delivery model train/test split
- temporal churn snapshot with a future 60-day label window
- model comparison and persisted artifacts
- automated EDA HTML report
- Streamlit BI/ML application
- FastAPI scoring endpoints
- tests and Docker configuration

## 2. Important source-data findings handled by the code

The supplied data is intentionally/structurally dirty. The code does **not** fabricate corrections where the correct value is unknowable.

### Mixed dates

A numeric value such as `03/04/2024` is inherently ambiguous. `src/utils.py` applies the configurable policy in `config/config.yaml` and retains an `AmbiguousFlag`, so the assumption stays auditable.

### Cross-city entity relationships

Many orders connect a customer, restaurant and delivery partner from different cities. Those relationships cannot be safely regenerated from the available columns. They are retained and explicitly flagged in the analytical feature table.

### Financial inconsistency

The raw order-level food cost does not reconcile with order items for most orders. The cleaned pipeline treats line-item totals as the auditable food-cost source, recomputes `Quantity × UnitPrice`, and reconstructs `FinalAmount` when the required components are valid.

### Delivery prediction limitation

The current synthetic source target has very weak observable signal. The included modeling code still follows a correct production workflow and compares against a mean baseline, but it reports poor source-data predictability rather than forcing an unrealistic MAE target.

### Churn limitation

Customers have sparse order histories and the generated churn outcome is highly imbalanced. Churn is therefore constructed correctly as an observation-window / future-window problem, but model metrics must be interpreted as a data-generation limitation.

## 3. Folder structure

```text
zomato_advanced_project/
├── app/
│   └── streamlit_app.py
├── api/
│   └── main.py
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
├── models/
├── reports/
├── sql/
│   ├── 00_original_schema.sql
│   ├── 00_original_business_queries.sql
│   ├── 01_schema_clean.sql
│   ├── 02_analytics_views.sql
│   ├── 03_business_queries_corrected.sql
│   └── 04_powerbi_measures.dax
├── src/
│   ├── cleaning.py
│   ├── db.py
│   ├── eda.py
│   ├── features.py
│   ├── modeling.py
│   ├── pipeline.py
│   ├── scoring.py
│   ├── settings.py
│   ├── utils.py
│   └── validation.py
├── tests/
│   └── test_core.py
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
├── run_pipeline.py
└── README.md
```

## 4. Windows setup

Open Command Prompt or PowerShell inside the project folder.

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Then run the complete pipeline:

```powershell
python run_pipeline.py
```

For a faster data/EDA-only run:

```powershell
python run_pipeline.py --skip-models
```

## 5. Individual pipeline stages

```powershell
python -m src.cleaning
python -m src.validation
python -m src.features
python -m src.eda
python -m src.modeling
```

### Outputs

Cleaning creates cleaned CSV and Python-native `.pkl` files in `data/processed/`. Feature engineering writes model-ready tables to `data/features/`. Model reports, predictions and automated EDA outputs are placed in `reports/`. Trained scikit-learn pipelines are saved to `models/`.

## 6. Data-cleaning logic

`src/cleaning.py` performs the following:

1. removes duplicate primary keys;
2. normalizes city strings and known aliases;
3. parses mixed dates while retaining ambiguity/parse flags;
4. converts invalid ages, ratings and negative monetary values to missing values rather than inventing replacements;
5. swaps obviously reversed promotion start/end dates while flagging the repair;
6. repairs invalid item unit price from valid menu price where defensible;
7. drops impossible non-positive quantities because quantity cannot be inferred;
8. recomputes order-item `TotalPrice`;
9. drops true foreign-key orphans;
10. rebuilds `FoodCost` from valid order items;
11. rebuilds `FinalAmount` from `FoodCost + DeliveryFee - Discount + GST` when financially valid;
12. recomputes each customer's actual `TotalOrders` from the cleaned order fact table.

The detailed audit is written to:

```text
data/processed/data_quality_report.json
data/processed/data_quality_summary.csv
```

## 7. Validation

Run:

```powershell
python -m src.validation
```

The validator checks primary-key uniqueness, foreign keys and the financial equation. Structural source problems such as cross-city relationships and ambiguous dates are written as warnings rather than being hidden.

Output:

```text
reports/validation_report.json
```

## 8. Delivery-time feature engineering

`src/features.py` creates an order-level training table using only attributes available at or before the order outcome. It includes:

- basket quantity/value and item-line features
- weighted preparation time
- customer membership and tenure
- restaurant cuisine/rating/cost/type
- partner vehicle/rating/tenure/history
- hour, weekday, month, lunch/dinner/weekend flags
- coupon and discount features
- daily weather at the restaurant city
- nearest traffic observation within a configurable time tolerance
- customer/restaurant/partner cross-city mismatch indicators
- leakage-safe historical order count and prior mean delivery-time features

`DistanceKm` is automatically activated if future source data contains delivery latitude/longitude. It is deliberately not fabricated for the current source.

Output:

```text
data/features/delivery_features.pkl
data/features/delivery_features.csv
```

## 9. Delivery model

Run:

```powershell
python -m src.modeling
```

Regression candidates:

- Ridge regression
- Random Forest
- Extra Trees

The split is **chronological**, not random. This prevents future orders leaking into historical evaluation.

Metrics:

- MAE
- RMSE
- R²
- mean-prediction baseline comparison

Artifacts:

```text
models/delivery_model.joblib
reports/delivery_model_report.json
reports/delivery_test_predictions.csv
```

The supplied dataset currently produces model performance close to or worse than a mean baseline. That result is itself an important analytical finding: additional causal data or regenerated synthetic relationships are required before the PRD target of roughly MAE < 5 minutes is defensible.

## 10. Churn definition and model

A churn label must not be derived from the same time period used to build the features. The project therefore uses:

```text
historical data <= snapshot date       -> features
snapshot date + 1 ... + 60 days        -> outcome window
no order in future window              -> Churn60d = 1
```

Features include RFM-style variables, cancellation/late rate, historical spend, average order value, order velocity, restaurant/payment diversity, customer tenure and membership/cuisine attributes.

Classification candidates:

- Logistic Regression
- Random Forest
- Extra Trees

The code uses separate train/validation/test sets. The validation set selects the probability threshold; the test set remains untouched until final reporting.

Artifacts:

```text
models/churn_model.joblib
reports/churn_model_report.json
reports/churn_test_predictions.csv
```

## 11. Automated EDA

```powershell
python -m src.eda
```

Open:

```text
reports/eda_report.html
```

It contains executive KPIs and interactive Plotly charts for revenue, order status, cities, delivery times and churn/recency.

## 12. Streamlit application

After running the pipeline:

```powershell
streamlit run app/streamlit_app.py
```

The application has five sections:

- Executive BI
- Data Quality
- Delivery ML
- Churn ML
- Scoring Demo

The scoring demo predicts an existing order/customer directly from the engineered feature store.

## 13. FastAPI service

Start the API:

```powershell
uvicorn api.main:app --reload
```

Then use:

```text
GET /health
GET /predict/delivery/{order_id}
GET /predict/churn/{customer_id}
```

Interactive Swagger documentation is available from FastAPI at `/docs` while the server is running.

## 14. PostgreSQL database

Start PostgreSQL yourself or use Docker:

```powershell
docker compose up -d postgres
```

Copy the example environment file:

```powershell
copy .env.example .env
```

Default Docker connection string:

```text
postgresql+psycopg2://zomato:zomato@localhost:5432/zomato
```

Load the cleaned relational database:

```powershell
python -m src.db
```

Then run:

```text
sql/02_analytics_views.sql
sql/03_business_queries_corrected.sql
```

## 15. Power BI implementation

Recommended model relationships:

```text
cities[City] 1-* customers[City]
cities[City] 1-* restaurants[City]
cities[City] 1-* delivery_partners[City]
customers[CustomerID] 1-* orders[CustomerID]
restaurants[RestaurantID] 1-* orders[RestaurantID]
delivery_partners[DeliveryPartnerID] 1-* orders[DeliveryPartnerID]
orders[OrderID] 1-* order_items[OrderID]
orders[OrderID] 1-* payments[OrderID]
orders[OrderID] 1-* customer_feedback[OrderID]
menu[FoodItemID] 1-* order_items[FoodItemID]
```

Create a proper Calendar table and relate it to `orders[OrderDate]`. Use `sql/04_powerbi_measures.dax` for the supplied DAX measures.

Suggested report pages:

1. Executive Overview
2. Revenue & Order Trends
3. Restaurant Performance
4. Delivery Operations
5. Customer & Churn
6. Promotions & Payments
7. Data Quality Audit

Do **not** combine customer city and restaurant city into one ambiguous city dimension in visual calculations without specifying which business question it represents. Because the synthetic source links them inconsistently, operational KPIs should generally use restaurant city.

## 16. Docker

Run PostgreSQL + dashboard + API:

```powershell
docker compose up --build
```

Services:

- Streamlit: port `8501`
- FastAPI: port `8000`
- PostgreSQL: port `5432`

The pipeline should be run once to create the processed feature/model artifacts before using the scoring UI/API.

## 17. Tests

```powershell
pytest -q
```

Tests cover mixed-date interpretation/ambiguity flags, primary-key uniqueness and financial reconciliation.

## 18. Recommended improvements to the underlying dataset

For a stronger final predictive project, regenerate or add:

- `PromisedDeliveryTimeMinutes` or a promised delivery timestamp
- order-level delivery latitude and longitude
- city-consistent customer/restaurant/partner allocation
- delivery time generated as a function of distance, preparation time, traffic, weather and partner workload
- longer customer histories with more repeat ordering
- a clearly defined data-generation timestamp to eliminate mixed-date ambiguity

Those changes would allow the ML success criteria in the PRD to measure actual model quality instead of source-data randomness.

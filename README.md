# Zomato Business Intelligence & Delivery Time Prediction Platform

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green)
![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-blue)
![Docker](https://img.shields.io/badge/Deployment-Docker-blue)

An end-to-end **Business Intelligence and Machine Learning platform** for analyzing Zomato food-delivery data, generating business insights, predicting delivery time, and identifying customer churn.

## 🚀 Key Features

- Data cleaning and validation
- Exploratory Data Analysis
- SQL-based business analytics
- Delivery-time prediction
- 60-day customer churn prediction
- Interactive Streamlit dashboard
- FastAPI prediction endpoints
- PostgreSQL integration
- Power BI measures
- Docker deployment
- Automated testing

## 🛠 Tech Stack

**Python | Pandas | NumPy | Scikit-learn | PostgreSQL | SQLAlchemy | Streamlit | FastAPI | Power BI | Docker | Plotly**

## 📊 Project Workflow

```text
Raw Data
   ↓
Data Cleaning & Validation
   ↓
Feature Engineering
   ↓
SQL & Business Analytics
   ↓
Exploratory Data Analysis
   ↓
Machine Learning
   ↓
Delivery Prediction + Churn Prediction
   ↓
Streamlit Dashboard / FastAPI / Power BI
```

## 📁 Project Structure

```text
Zomato Business/
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
│   ├── delivery_model.joblib
│   └── churn_model.joblib
├── reports/
├── sql/
├── src/
├── tests/
├── run_pipeline.py
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## ⚙️ Installation

Clone the repository and create a virtual environment:

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Project

Run the complete data and machine-learning pipeline:

```bash
python run_pipeline.py
```

Run the Streamlit dashboard:

```bash
streamlit run app/streamlit_app.py
```

Run the FastAPI application:

```bash
uvicorn api.main:app --reload
```

Open the API documentation at:

```text
http://localhost:8000/docs
```

## 🤖 Machine Learning Models

### Delivery Time Prediction
Predicts the expected delivery duration using order, restaurant, customer, delivery-partner, weather, traffic, and historical features.

Models evaluated include:

- Ridge Regression
- Random Forest
- Extra Trees

### Customer Churn Prediction
Identifies customers likely to become inactive within a **60-day future window** using historical customer behaviour.

## 📈 Business Intelligence

The platform analyzes:

- Orders and revenue
- Customer behaviour
- Restaurant performance
- Delivery efficiency
- Cancellation trends
- City-level performance
- Customer churn
- Operational trends

## 📊 Dashboard

The Streamlit application provides an interactive interface for exploring business KPIs, delivery performance, customer behaviour, and machine-learning predictions.

### Dashboard Preview

```text
Add your Streamlit dashboard screenshot here.

Example:
assets/dashboard.png
```

Then add:

```markdown
![Dashboard](assets/dashboard.png)
```

## 🔌 API Endpoints

```text
GET /health

GET /predict/delivery/{order_id}

GET /predict/churn/{customer_id}
```

## 🐳 Docker

Launch PostgreSQL, Streamlit, and FastAPI together:

```bash
docker compose up --build
```

Available services:

```text
Streamlit Dashboard → http://localhost:8501
FastAPI             → http://localhost:8000
PostgreSQL           → localhost:5432
```

## ✅ Project Results

- **20,377** cleaned orders
- **13,481** completed orders
- **21/21** integrity validation checks passed
- Delivery prediction model developed using chronological validation
- 60-day churn prediction pipeline implemented
- Streamlit, FastAPI, PostgreSQL, Power BI, and Docker integration completed

## 🎯 Objective

To demonstrate how **data analytics, business intelligence, machine learning, APIs, and dashboards** can be combined into a complete food-delivery analytics platform for data-driven decision-making.



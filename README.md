# Global Superstore Machine Learning

End-to-end machine learning project for **BNCC LnT Camp 2026** using the Global Superstore dataset.

The project covers data integration from a normalized SQLite database, exploratory data analysis, feature engineering, regression, customer segmentation, model persistence, REST API development, frontend integration, and cloud deployment.

## Live Application

**Streamlit App:**  
https://global-superstore-ml.streamlit.app

**GitHub Repository:**  
https://github.com/Tjiuaiching/global-superstore-ml

---

## Project Objectives

This project focuses on two machine learning tasks.

### 1. Regression — Profit Prediction

Predict the profit of an individual product line using:

- Discount
- Quantity
- Sales
- Shipping cost
- Product category
- Customer segment
- Ship mode
- Order priority
- Market

The final model uses a **Random Forest Regressor**.

### 2. Clustering — Customer Segmentation

Group customers based on purchasing behavior using:

- Total spending
- Order count
- Average order value
- Average discount

The final clustering model uses **K-Means with K = 5**.

---

## Dataset

The project uses the **Global Superstore Dataset**, restructured into a normalized SQLite database for the BNCC LnT Camp 2026 Final Project.

The original dataset is available on Kaggle:

https://www.kaggle.com/datasets/fatihilhan/global-superstore-dataset

The normalized SQLite database can be downloaded from:

https://drive.google.com/file/d/1M2sonY7serOCzWYDCKEdxZ5fJ7quOoKd/view?usp=sharing

The database itself is not stored in this repository.

More information is available in:

`notebook/DATASET.md`

---

## Database Structure

The SQLite database contains the following main tables:

- `orders`
- `order_items`
- `dim_customers`
- `dim_products`
- `dim_locations`
- `dim_product_name_variants`

SQL JOIN queries are used to assemble the final working dataset.

The resulting machine learning dataset contains:

- **51,290 product-line records**
- **26 original joined features**

---

## Exploratory Data Analysis

Several business patterns were identified during EDA:

- Technology generated the highest total profit among product categories.
- Furniture had a noticeably lower profit margin than Technology and Office Supplies.
- Higher discount levels were generally associated with lower profitability.
- APAC generated the highest total profit among markets.
- Canada showed a high profit margin, although it had a much smaller product-line volume.
- Sales and profit increased across the observed yearly period.
- Discount showed a negative relationship with profit, while sales showed a positive relationship with profit.

---

## Regression Results

Two regression approaches were evaluated.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 58.58 | 138.69 | 0.357 |
| Random Forest | 34.58 | 86.62 | 0.749 |

The Random Forest model performed substantially better and was selected as the final regression approach.

The production pipeline achieved approximately:

- **MAE:** 34.62
- **RMSE:** 87.31
- **R²:** 0.745

---

## Customer Segmentation

K-Means clustering was evaluated using both the **Elbow Method** and **Silhouette Score**.

Although K = 2 produced the highest Silhouette Score at approximately **0.3336**, K = 5 achieved a very similar score of approximately **0.3300** while providing more detailed and actionable customer groups.

The final customer segments are:

1. **High-Ticket Customers**
2. **Frequent Regular Customers**
3. **Low-Value Occasional Customers**
4. **High-Discount Unprofitable Customers**
5. **High-Value Frequent Customers**

These segments can support customer retention, cross-selling, re-engagement, and discount optimization strategies.

---

## Project Architecture

```text
User
 │
 ▼
Streamlit Frontend
 │
 │ HTTP / JSON
 ▼
FastAPI Backend
 │
 ├── Regression Pipeline
 │
 └── Clustering Pipeline
 │
 ▼
Saved Joblib Models
```

The frontend does not load the machine learning models directly. Instead, it sends user input to the FastAPI backend through HTTP requests. The backend loads the saved model artifacts, performs the prediction or clustering process, and returns the result to the frontend.

---

## Repository Structure

```text
global-superstore-ml/
│
├── notebook/
│   ├── Global_Superstore_ML_Final_Project.ipynb
│   └── DATASET.md
│
├── model/
│   ├── regression_pipeline.pkl
│   ├── clustering_pipeline.pkl
│   └── cluster_labels.pkl
│
├── backend/
│   ├── main.py
│   └── requirements.txt
│
├── frontend/
│   ├── app.py
│   └── requirements.txt
│
├── app.py
├── requirements.txt
├── .python-version
├── .gitignore
└── README.md
```

The repository follows a monorepo structure that separates the machine learning notebook, saved model artifacts, backend API, and frontend application while keeping all components in a single project repository.

---

## Backend API

The backend is built using **FastAPI** and loads the saved machine learning pipelines from the `model/` directory.

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "regression_model": "loaded",
  "clustering_model": "loaded"
}
```

### Profit Prediction

```http
POST /predict/regression
```

Example input:

```json
{
  "discount": 0.1,
  "quantity": 3,
  "sales": 500,
  "shipping_cost": 25,
  "category": "Technology",
  "segment": "Consumer",
  "ship_mode": "Standard Class",
  "order_priority": "Medium",
  "market": "APAC"
}
```

Example output:

```json
{
  "predicted_profit": 21.24
}
```

### Customer Segmentation

```http
POST /predict/clustering
```

Example input:

```json
{
  "total_spending": 5000,
  "order_count": 5,
  "average_order_value": 1000,
  "average_discount": 0.1
}
```

Example output:

```json
{
  "cluster": 0,
  "segment": "High-Ticket Customers"
}
```

---

## Running the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/Tjiuaiching/global-superstore-ml.git
cd global-superstore-ml
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Backend Dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Start the FastAPI Backend

Run the following command from the project root:

```bash
uvicorn backend.main:app --reload
```

The backend will run locally at:

```text
http://127.0.0.1:8000
```

FastAPI interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

### 5. Install Frontend Dependencies

```bash
pip install -r frontend/requirements.txt
```

### 6. Start the Streamlit Frontend

```bash
streamlit run frontend/app.py
```

When running locally, the frontend uses:

```text
http://127.0.0.1:8000
```

as the default FastAPI backend.

---

## Deployment

The application is deployed using:

- **Backend:** FastAPI on Vercel
- **Frontend:** Streamlit Community Cloud
- **Frontend URL:** https://global-superstore-ml.streamlit.app
- **Model Storage:** Joblib serialized Scikit-learn pipelines

The regression pipeline stored in the repository uses **LZ4 compression** to reduce its file size. Compression only changes how the trained model is stored and does not retrain or modify the model or its predictions.

---

## Technology Stack

- Python
- SQLite
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- FastAPI
- Pydantic
- Streamlit
- Vercel
- Streamlit Community Cloud
- Git
- GitHub

---

## Notebook

The complete machine learning workflow is available in:

`notebook/Global_Superstore_ML_Final_Project.ipynb`

The notebook includes:

- Problem statement and objectives
- Database connection and SQL JOINs
- Exploratory data analysis
- Data preprocessing and feature engineering
- Regression modeling and evaluation
- Customer segmentation using K-Means
- Business interpretation
- Production pipelines
- Model persistence
- Conclusion and business insights

Dataset access instructions are available in:

`notebook/DATASET.md`

---

## Author

**Carolina Lestari Sutjipto**

Computer Science  
BINUS University

BNCC LnT Camp 2026 — Machine Learning Final Project
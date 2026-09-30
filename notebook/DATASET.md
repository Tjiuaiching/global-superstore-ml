# Dataset Access

This project uses the Global Superstore Dataset, which has been restructured into a normalized SQLite database for the BNCC LnT Camp 2026 Final Project.

## Original Dataset

The original Global Superstore dataset is available on Kaggle:

https://www.kaggle.com/datasets/fatihilhan/global-superstore-dataset

## SQLite Database

For this project, the dataset is provided as a normalized SQLite database named:

`superstore.sqlite`

The database can be downloaded from:

https://drive.google.com/file/d/1M2sonY7serOCzWYDCKEdxZ5fJ7quOoKd/view?usp=sharing

The SQLite database is not included directly in this repository.

## Database Tables

The database contains the following main tables:

- `orders`
- `order_items`
- `dim_customers`
- `dim_products`
- `dim_locations`
- `dim_product_name_variants`

The notebook connects to the database using Python's `sqlite3` library and uses SQL JOIN queries to construct the working dataset.

## Running the Notebook

1. Download `superstore.sqlite` using the link above.
2. Open `Global_Superstore_ML_Final_Project.ipynb` in Google Colab.
3. Run the notebook from top to bottom.
4. When prompted by the upload cell, upload `superstore.sqlite`.
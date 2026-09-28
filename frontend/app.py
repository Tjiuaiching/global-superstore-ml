import streamlit as st
import requests
import os

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)

st.set_page_config(
    page_title="Global Superstore ML",
    page_icon="📊",
    layout="wide"
)

st.title("Global Superstore ML")

st.write(
    "Predict transaction profit and analyze customer segments using machine learning."
)

with st.sidebar:
    st.header("About")
    st.write(
        "An end-to-end machine learning application using "
        "Random Forest Regression and K-Means Clustering."
    )

    st.caption("Backend: FastAPI")
    st.caption("Frontend: Streamlit")
    st.caption("ML: scikit-learn")

    try:
        health = requests.get(
            f"{API_URL}/health",
            timeout=5
        )

        if health.status_code == 200:
            st.success("API Connected")
        else:
            st.warning("API Unavailable")

    except requests.exceptions.RequestException:
        st.error("API Disconnected")

st.subheader("Profit Prediction")

st.caption(
    "Estimate the expected profit of a transaction based on "
    "sales, discount, shipping, customer, and order information."
)

with st.form("regression_form"):
    col1, col2 = st.columns(2)

    with col1:
        sales = st.number_input(
            "Sales",
            min_value=0.0,
            value=500.0
        )

        discount = st.number_input(
            "Discount",
            min_value=0.0,
            max_value=1.0,
            value=0.2
        )

        quantity = st.number_input(
            "Quantity",
            min_value=1,
            value=3
        )

        shipping_cost = st.number_input(
            "Shipping Cost",
            min_value=0.0,
            value=25.0
        )

    with col2:
        category = st.selectbox(
            "Category",
            ["Furniture", "Office Supplies", "Technology"]
        )

        segment = st.selectbox(
            "Customer Segment",
            ["Consumer", "Corporate", "Home Office"]
        )

        ship_mode = st.selectbox(
            "Ship Mode",
            [
                "Standard Class",
                "Second Class",
                "First Class",
                "Same Day"
            ]
        )

        order_priority = st.selectbox(
            "Order Priority",
            ["Low", "Medium", "High", "Critical"]
        )

        market = st.selectbox(
            "Market",
            ["APAC", "Africa", "Canada", "EMEA", "EU", "LATAM", "US"]
        )

    submit_regression = st.form_submit_button(
        "Predict Profit"
    )

if submit_regression:
    payload = {
        "discount": discount,
        "quantity": quantity,
        "sales": sales,
        "shipping_cost": shipping_cost,
        "category": category,
        "segment": segment,
        "ship_mode": ship_mode,
        "order_priority": order_priority,
        "market": market
    }

    try:
        response = requests.post(
            f"{API_URL}/predict/regression",
            json=payload,
            timeout=30
        )

        response.raise_for_status()

        result = response.json()
        predicted_profit = result["predicted_profit"]

        st.success("Prediction completed successfully!")
        st.metric(
            "Predicted Profit",
            f"${predicted_profit:,.2f}"
        )

    except requests.exceptions.RequestException:
        st.error(
            "Could not connect to the backend. "
            "Make sure FastAPI is running on port 8000."
        )

st.divider()
st.subheader("Customer Segmentation")

st.caption(
    "Identify a customer's purchasing segment based on "
    "spending behaviour, order frequency, and discount usage."
)

with st.form("clustering_form"):
    total_spending = st.number_input(
        "Total Spending",
        min_value=0.0,
        value=5000.0
    )

    order_count = st.number_input(
        "Number of Orders",
        min_value=1,
        value=5
    )

    average_order_value = st.number_input(
        "Average Order Value",
        min_value=0.0,
        value=1000.0
    )

    average_discount = st.number_input(
        "Average Discount",
        min_value=0.0,
        max_value=1.0,
        value=0.1
    )

    submit_clustering = st.form_submit_button(
        "Analyze Customer"
    )

segment_descriptions = {
    "High-Ticket Customers":
        "Customers who purchase less frequently but spend a high amount per order.",

    "Frequent Regular Customers":
        "Customers who purchase relatively often with moderate spending.",

    "Low-Value Occasional Customers":
        "Customers with relatively low spending and low purchase frequency.",

    "High-Discount Unprofitable Customers":
        "Customers associated with high discount usage and lower profitability.",

    "High-Value Frequent Customers":
        "Customers with high spending, frequent purchases, and strong profitability."
}

if submit_clustering:
    payload = {
        "total_spending": total_spending,
        "order_count": order_count,
        "average_order_value": average_order_value,
        "average_discount": average_discount
    }

    try:
        response = requests.post(
            f"{API_URL}/predict/clustering",
            json=payload,
            timeout=30
        )

        response.raise_for_status()

        result = response.json()

        st.success("Customer segmentation completed!")
        st.metric(
            "Customer Segment",
            result["segment"]
        )

        st.write(
        segment_descriptions.get(
            result["segment"],
            "Customer segment identified by the clustering model."
            )
        )
        
        st.caption(
            f"Cluster {result['cluster']}"
        )

    except requests.exceptions.RequestException:
        st.error(
            "Could not connect to the backend. "
            "Make sure FastAPI is running on port 8000."
        )
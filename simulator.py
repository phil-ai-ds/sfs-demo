import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Load data from CSV files
@st.cache_data  # Use st.cache_data for loading data
def load_data():
    initial_stock = pd.read_csv("initial_stock.csv")
    daily_demand = pd.read_csv("daily_demand.csv")
    shipment_plan = pd.read_csv("shipment_plan.csv")
    return initial_stock, daily_demand, shipment_plan

# Simulator logic
def calculate_safety_stock(demand_data, safety_days_by_category, product_to_category):
    """
    Calculate safety stock for each store, product, and day based on average past 7 days' sales.
    """
    safety_stock = demand_data.copy()
    safety_stock['Avg_Sales_7_Days'] = safety_stock.groupby(['Location', 'Product'])['Demand'].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    safety_stock['Category'] = safety_stock['Product'].map(product_to_category)
    safety_stock['Safety_Stock'] = safety_stock.apply(
        lambda row: row['Avg_Sales_7_Days'] * safety_days_by_category.get(row['Category'], 0),
        axis=1
    )
    return safety_stock[['Day', 'Location', 'Product', 'Safety_Stock']]

def simulate_fulfillment(initial_stock, daily_demand, safety_data, reservation_ratios_by_category, product_to_category, offline_stores):
    """
    Optimized simulation of fulfillment rates based on safety stock and reservation ratios, with robust handling of shared stock.
    """
    # Prepare stock data for simulation
    stock_data = initial_stock.copy()
    stock_data.rename(columns={"Initial_Stock": "Remaining_Stock"}, inplace=True)

    # Add fulfillment tracking columns
    daily_demand = daily_demand.copy()
    daily_demand["Fulfilled"] = 0
    daily_demand["Unfulfilled"] = daily_demand["Demand"]  # At first, all demand is unfulfilled
    daily_demand["Shared_Stock_To_Online"] = 0

    # Map products to their categories
    daily_demand["Category"] = daily_demand["Product"].map(product_to_category)

    # Loop through each day
    for day in range(1, daily_demand["Day"].max() + 1):
        # Get shared stock for all stores and products for the current day
        daily_safety_stock = safety_data[safety_data["Day"] == day][["Location", "Product", "Safety_Stock"]]
        stock_data = stock_data.merge(daily_safety_stock, on=["Location", "Product"], how="left")
        stock_data["Safety_Stock"].fillna(0, inplace=True)

        # Calculate shared stock
        stock_data["Shared_Stock"] = (stock_data["Remaining_Stock"] - stock_data["Safety_Stock"]).clip(lower=0)
        stock_data["Shared_Stock"] *= stock_data["Product"].map(
            lambda p: reservation_ratios_by_category.get(product_to_category[p], 0.0)
        )
        stock_data["Shared_Stock"] = stock_data["Shared_Stock"].clip(lower=0)  # Ensure no negative shared stock

        # Summarize shared stock available for each product
        shared_stock_data = stock_data[stock_data["Location"].isin(offline_stores)]
        shared_stock_summary = shared_stock_data.groupby("Product")["Shared_Stock"].sum().reset_index()

        # Distribute shared stock to online demand
        online_mask = (daily_demand["Day"] == day) & (daily_demand["Location"] == "Online Platform")
        online_demand = daily_demand.loc[online_mask]
        
        # Merge shared stock summary to online demand
        online_demand = online_demand.merge(shared_stock_summary, on="Product", how="left", suffixes=("", "_Available"))
        online_demand["Shared_Stock_Available"].fillna(0, inplace=True)

        # Calculate fulfilled demand for online
        online_demand["Shared_Stock_To_Online"] = online_demand[["Shared_Stock_Available", "Unfulfilled"]].min(axis=1)
        online_demand["Fulfilled"] += online_demand["Shared_Stock_To_Online"]
        online_demand["Unfulfilled"] -= online_demand["Shared_Stock_To_Online"]

        # Summarize shared stock used by online into a new DataFrame
        usage_summary = online_demand.groupby("Product")["Shared_Stock_To_Online"].sum().reset_index()
        usage_summary.rename(columns={"Shared_Stock_To_Online": "Shared_Stock_Used"}, inplace=True)

        # Merge usage_summary into stock_data
        stock_data = stock_data.merge(usage_summary, on="Product", how="left")
        stock_data["Shared_Stock_Used"].fillna(0, inplace=True)  # Ensure no NaNs in the used column

        # Update shared stock after online usage
        stock_data["Shared_Stock"] -= stock_data["Shared_Stock_Used"]

        # Fulfill offline demand
        offline_mask = (daily_demand["Day"] == day) & (daily_demand["Location"] != "Online Platform")
        offline_demand = daily_demand.loc[offline_mask]

        # Merge offline demand with stock data
        stock_data_offline = stock_data[stock_data["Location"].isin(offline_stores)]
        offline_demand = offline_demand.merge(
            stock_data_offline[["Location", "Product", "Remaining_Stock"]],
            on=["Location", "Product"],
            how="left",
        )
        offline_demand["Remaining_Stock"].fillna(0, inplace=True)

        # Calculate fulfillment
        offline_demand["Fulfillable"] = offline_demand[["Remaining_Stock", "Unfulfilled"]].min(axis=1)
        offline_demand["Fulfilled"] += offline_demand["Fulfillable"]
        offline_demand["Unfulfilled"] -= offline_demand["Fulfillable"]

        # Update stock based on offline fulfillment
        stock_data_offline = stock_data_offline.merge(
            offline_demand.groupby(["Location", "Product"])["Fulfillable"].sum().reset_index(),
            on=["Location", "Product"],
            how="left",
        )
        stock_data_offline["Fulfillable"].fillna(0, inplace=True)
        stock_data_offline["Remaining_Stock"] -= stock_data_offline["Fulfillable"]

        # Combine updated offline stock back into stock_data
        stock_data = pd.concat([stock_data_offline, stock_data[stock_data["Location"] == "Online Platform"]])

        # Update daily_demand DataFrame
        daily_demand.update(online_demand)
        daily_demand.update(offline_demand)

    return daily_demand


def calculate_fulfillment_rates(demand_data):
    """
    Calculate fulfillment rates by store and product.
    """
    fulfillment_rates = demand_data.groupby(['Location', 'Product']).agg(
        Total_Demand=('Demand', 'sum'),
        Total_Fulfilled=('Fulfilled', 'sum')
    )
    fulfillment_rates['Fulfillment_Rate'] = fulfillment_rates['Total_Fulfilled'] / fulfillment_rates['Total_Demand']
    return fulfillment_rates.reset_index()

# Set up the Streamlit webpage
st.title("Ship-from-Store Strategy Simulator")
st.sidebar.header("Simulation Parameters")

# Load data
initial_stock, daily_demand, shipment_plan = load_data()
products = daily_demand['Product'].unique()
locations = daily_demand['Location'].unique()

# Separate offline stores
offline_stores = [loc for loc in locations if loc != "Online Platform"]

# Categorize products into 3 groups
categories = {
    "Category_A": products[:len(products) // 3],
    "Category_B": products[len(products) // 3: 2 * len(products) // 3],
    "Category_C": products[2 * len(products) // 3:]
}
product_to_category = {product: category for category, prods in categories.items() for product in prods}

# User inputs for safety days and reservation ratios by category
st.sidebar.subheader("Safety Days by Category")
safety_days_by_category = {
    category: st.sidebar.slider(f"{category} Safety Days", 0, 14, 7) for category in categories.keys()
}
st.sidebar.subheader("Reservation Ratios by Category")
reservation_ratios_by_category = {
    category: st.sidebar.slider(f"{category} Reservation Ratio", 0.0, 1.0, 0.5) for category in categories.keys()
}

# Calculate safety stock
safety_stock = calculate_safety_stock(daily_demand, safety_days_by_category, product_to_category)

# Simulate fulfillment
simulated_demand = simulate_fulfillment(initial_stock, daily_demand, safety_stock, reservation_ratios_by_category, product_to_category, offline_stores)

# Calculate fulfillment rates
fulfillment_rates = calculate_fulfillment_rates(simulated_demand)

# Prepare heatmap data
heatmap_data = fulfillment_rates.pivot(
    index='Location', columns='Product', values='Fulfillment_Rate'
).fillna(0)

# Visualization: Heatmap
st.header("Fulfillment Rates Heatmap")
fig = px.imshow(
    heatmap_data,
    text_auto=True,
    aspect="auto",
    title="Fulfillment Rates by Store/Platform and Product",
    labels=dict(x="Product", y="Store/Platform", color="Fulfillment Rate"),
    color_continuous_scale='Blues'
)
st.plotly_chart(fig)

# Store and Product Selection
# Store and Product Selection
st.header("Daily Stock, Demand, and Performance Metrics")
selected_stores = st.multiselect("Select Stores", locations, default=offline_stores)
selected_products = st.multiselect("Select Products", products, default=products[:3])

# Filter data for selected stores and products
filtered_data = simulated_demand[
    (simulated_demand['Location'].isin(selected_stores)) &
    (simulated_demand['Product'].isin(selected_products))
]

# Plot daily metrics using Plotly
st.subheader("Daily Stock Levels, Demand, Shared Stock, and Fulfillment Rates")
if not filtered_data.empty:
    # Reshape filtered_data into a long format for multi-metric plots
    melted_data = filtered_data.melt(
        id_vars=["Day", "Location", "Product"],  # Columns to keep
        value_vars=["Remaining_Stock", "Demand", "Shared_Stock_To_Online", "Fulfilled"],  # Columns to unpivot
        var_name="Metric",  # New column for the metric name
        value_name="Value"  # New column for the metric value
    )

    # Create a line plot with the reshaped data
    fig = px.line(
        melted_data,
        x="Day",
        y="Value",
        color="Location",
        line_group="Metric",
        facet_row="Metric",  # Separate lines by metric type
        labels={"Value": "Value", "Day": "Day", "Metric": "Metric Type"},
        title="Daily Metrics (Stock, Demand, Shared Stock, Fulfillment)"
    )
    st.plotly_chart(fig)
else:
    st.write("No data available for the selected stores and products.")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Load data from CSV files
@st.cache
def load_data():
    initial_stock = pd.read_csv("initial_stock.csv")
    daily_demand = pd.read_csv("daily_demand.csv")
    shipment_plan = pd.read_csv("shipment_plan.csv")
    return initial_stock, daily_demand, shipment_plan

# Simulator logic
def calculate_safety_stock(demand_data, safety_days_by_category, product_to_category):
    """
    Calculate safety stock for each store, product, and day based on average past 7 days' sales and categories.
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
    Simulate the fulfillment rates based on safety stock and reservation ratios.
    """
    # Prepare stock data for simulation
    stock_data = initial_stock.copy()
    stock_data.rename(columns={'Initial_Stock': 'Remaining_Stock'}, inplace=True)

    # Add fulfillment tracking columns
    daily_demand['Fulfilled'] = 0  # How much demand is fulfilled
    daily_demand['Unfulfilled'] = 0  # How much demand is not fulfilled

    # Iterate through days to simulate demand fulfillment
    for day in range(1, daily_demand['Day'].max() + 1):
        # Update shared stock for the day and reset it by store, product
        shared_stock = {}

        # Calculate today's shared stock based on reservation ratios
        for idx, row in stock_data.iterrows():
            if row['Location'] != 'Online Platform':  # Only stores offer to share stock
                product = row['Product']
                store = row['Location']
                category = product_to_category[product]
                remaining_stock = row['Remaining_Stock']

                # Get safety stock for the store/product/day
                safety_stock_row = safety_data[
                    (safety_data['Day'] == day) &
                    (safety_data['Location'] == store) &
                    (safety_data['Product'] == product)
                ]
                safety_stock = safety_stock_row['Safety_Stock'].iloc[0] if not safety_stock_row.empty else 0

                # Calculate shared stock
                reserv_ratio = reservation_ratios_by_category.get(category, 0.0)
                excess_stock = max(remaining_stock - safety_stock, 0)
                shared_stock[(store, product)] = excess_stock * reserv_ratio

        # Fulfill demand for each location (including stores and online platform)
        for idx, row in daily_demand[daily_demand['Day'] == day].iterrows():
            location = row['Location']
            product = row['Product']
            demand = row['Demand']
            category = product_to_category[product]

            # Stock allocation rules
            if location == 'Online Platform':  # Online can only use its own stock + shared stock
                online_row = stock_data[
                    (stock_data['Location'] == 'Online Platform') & (stock_data['Product'] == product)
                ]
                remaining_online_stock = online_row['Remaining_Stock'].iloc[0] if not online_row.empty else 0

                # Fulfill using online stock first
                fulfilled = min(demand, remaining_online_stock)
                remaining_demand = demand - fulfilled

                # Then use shared stock
                for store in offline_stores:
                    if remaining_demand <= 0:
                        break
                    shared = shared_stock.get((store, product), 0)
                    additional_fulfilled = min(remaining_demand, shared)
                    fulfilled += additional_fulfilled
                    shared_stock[(store, product)] -= additional_fulfilled
                    remaining_demand -= additional_fulfilled
            else:  # Offline stores fulfill their own stock first and use shared stock if needed
                store_row = stock_data[(stock_data['Location'] == location) & (stock_data['Product'] == product)]
                remaining_store_stock = store_row['Remaining_Stock'].iloc[0] if not store_row.empty else 0

                # Fulfill from store's own stock
                fulfilled = min(demand, remaining_store_stock)
                remaining_demand = demand - fulfilled

                # Then use shared stock
                shared = shared_stock.get((location, product), 0)
                additional_fulfilled = min(remaining_demand, shared)
                fulfilled += additional_fulfilled
                shared_stock[(location, product)] -= additional_fulfilled

            # Update demand data for fulfillment
            daily_demand.loc[idx, 'Fulfilled'] = fulfilled
            daily_demand.loc[idx, 'Unfulfilled'] = demand - fulfilled

            # Update stock levels
            if location == 'Online Platform':
                stock_data.loc[
                    (stock_data['Location'] == 'Online Platform') & (stock_data['Product'] == product),
                    'Remaining_Stock'
                ] -= fulfilled
            else:
                stock_data.loc[
                    (stock_data['Location'] == location) & (stock_data['Product'] == product),
                    'Remaining_Stock'
                ] -= min(fulfilled, remaining_store_stock)

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

# Visualization
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

st.header("Simulated Fulfillment Details (Daily)")
st.dataframe(simulated_demand)
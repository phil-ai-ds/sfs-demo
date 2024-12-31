import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Load Data
initial_stock_df = pd.read_csv('initial_stock.csv')
daily_demand_df = pd.read_csv('daily_demand.csv')
shipment_plan_df = pd.read_csv('shipment_plan.csv')

# Split products into categories for simplicity
categories = {
    'Category_A': ['Product_1', 'Product_2', 'Product_3', 'Product_4', 'Product_5', 'Product_6', 'Product_7'],
    'Category_B': ['Product_8', 'Product_9', 'Product_10', 'Product_11', 'Product_12', 'Product_13'],
    'Category_C': ['Product_14', 'Product_15', 'Product_16', 'Product_17', 'Product_18', 'Product_19', 'Product_20']
}

# Initialize Streamlit sidebar inputs
safety_days = st.sidebar.slider("Safety Days", 0, 14, 3)  # Default safety days set to 3
reservation_ratios = {category: st.sidebar.slider(f"Reservation Ratio for {category}", 0.0, 1.0, 0.5) for category in categories}

# Function to compute safety stock for each day
def compute_safety_stock(daily_demand, safety_days, product_cats):
    safety_stock = {}
    for category, products in product_cats.items():
        for product in products:
            product_demand = daily_demand[daily_demand['Product'] == product].copy()
            # Calculate rolling 7-day average demand
            product_demand['Avg_Demand'] = product_demand['Demand'].rolling(window=7, min_periods=1).mean()
            product_demand['Safety_Stock'] = product_demand['Avg_Demand'] * safety_days
            # Collect results
            for _, row in product_demand.iterrows():
                safety_stock[(row['Day'], row['Location'], product)] = row['Safety_Stock']
    return safety_stock

# Compute safety stock
safety_stock = compute_safety_stock(daily_demand_df, safety_days, categories)

# Function to simulate order fulfillment
def simulate_order_fulfillment(initial_stock, demand, shipment_plan, safety_days, reservation_ratios, categories):
    fulfillment_records = []

    for (day, location), daily_group in demand.groupby(['Day', 'Location']):
        for category, products in categories.items():
            reservation_ratio = reservation_ratios[category]
            for product in products:
                # Extract relevant data
                product_demand = daily_group[daily_group['Product'] == product]['Demand'].sum()
                initial_stock_level = initial_stock.loc[
                    (initial_stock['Location'] == location) & (initial_stock['Product'] == product),
                    'Initial_Stock'].values[0]
                
                # Calculate safety stock
                safety_stock_level = safety_stock.get((day, location, product), 0)
                
                # Compute available stock for order
                shared_stock = max(0, (initial_stock_level - safety_stock_level) * reservation_ratio)
                local_stock = initial_stock_level - safety_stock_level - shared_stock
                
                # Calculate fulfilled orders
                if location == 'Online Platform':
                    fulfilled_orders = min(shared_stock, product_demand)
                else:
                    fulfilled_orders = min(local_stock + shared_stock, product_demand)
                
                fulfillment_rate = fulfilled_orders / product_demand if product_demand != 0 else 1
                
                # Track fulfillment record
                fulfillment_records.append({
                    'Day': day, 'Location': location, 'Product': product,
                    'Fulfillment_Rate': fulfillment_rate, 'Stock_Level': initial_stock_level,
                    'Demand': product_demand, 'Shared_Stock': shared_stock
                })

    return pd.DataFrame(fulfillment_records)

# Run simulation
fulfillment_df = simulate_order_fulfillment(initial_stock_df, daily_demand_df, shipment_plan_df, safety_days, reservation_ratios, categories)

# Calculate total fulfillment rate by product by store/platform
# Calculate total fulfillment rate by product by store/platform
total_fulfillment = fulfillment_df.groupby(['Location', 'Product'])['Fulfillment_Rate'].mean().reset_index()

# Proper use of pivot_table
pivot_table = total_fulfillment.pivot_table(index='Product', columns='Location', values='Fulfillment_Rate', fill_value=0)

# Heatmap Visualization using Plotly
heatmap_fig = px.imshow(
    pivot_table,
    labels=dict(x="Store/Platform", y="Product", color="Fulfillment Rate"),
    x=pivot_table.columns,
    y=pivot_table.index
)

st.title("Inventory Fulfillment Simulation")
st.header("Total Fulfillment Rate Heatmap")
st.plotly_chart(heatmap_fig)

# Multichoice widget
selected_stores = st.multiselect("Select Stores for Detail View", options=['Online Platform'] + [f'Store_{i}' for i in range(1, 11)], default=['Store_1'])
selected_products = st.multiselect("Select Products for Detail View", options=fulfillment_df['Product'].unique(), default=fulfillment_df['Product'].unique()[:3])

# Filter fulfillment_df for selected stores and products
filtered_fulfillment_df = fulfillment_df[(fulfillment_df['Location'].isin(selected_stores)) & (fulfillment_df['Product'].isin(selected_products))]

# Detailed visualization
for store in selected_stores:
    for product in selected_products:
        store_product_data = filtered_fulfillment_df[(filtered_fulfillment_df['Location'] == store) & (filtered_fulfillment_df['Product'] == product)]
        
        if not store_product_data.empty:
            fig = px.line(
                store_product_data, x='Day', y=['Stock_Level', 'Demand', 'Shared_Stock', 'Fulfillment_Rate'],
                title=f'Stock and Demand Details for {store} - {product}',
                labels={'value': 'Level/Rate', 'variable': 'Metric', 'Day': 'Day'},
                markers=True
            )
            st.plotly_chart(fig)

st.sidebar.markdown("### Configure Safety Days and Reservation Ratio")
st.sidebar.markdown("Modify safety days and reservation ratio to observe the impact on order fulfillment.")
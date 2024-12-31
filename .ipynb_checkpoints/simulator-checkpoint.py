import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Load the data
stock_data = pd.read_csv('initial_stock.csv')
demand_data = pd.read_csv('daily_demand.csv')
shipment_data = pd.read_csv('shipment_plan.csv')

# Set the number of days for the simulation
num_days = 30

# Initialize products categories
category_map = {product: f"Category_{(i % 3) + 1}" for i, product in enumerate(stock_data['Product'].unique())}

# Add category information
stock_data['Category'] = stock_data['Product'].map(category_map)
demand_data['Category'] = demand_data['Product'].map(category_map)
shipment_data['Category'] = shipment_data['Product'].map(category_map)

# Get unique stores and products
stores = stock_data['Location'].unique().tolist()
products = stock_data['Product'].unique().tolist()
categories = stock_data['Category'].unique().tolist()

# Sidebar for configuration
st.sidebar.header("Simulator Configuration")
category_safety_days = {cat: st.sidebar.number_input(f"{cat} Safety Days", min_value=0, max_value=10, value=2, key=f'safety_days_{cat}') for cat in categories}
category_reservation_ratio = {cat: st.sidebar.slider(f"{cat} Reservation Ratio", min_value=0.0, max_value=1.0, value=0.2, key=f'reservation_ratio_{cat}') for cat in categories}

# Calculate Safety Stock
def calculate_safety_stock(demand_data, safety_days_map):
    demand_data['Safety_Stock'] = 0.0

    for category, safety_days in safety_days_map.items():
        category_demand = demand_data[demand_data['Category'] == category].copy()
        rolling_avg_sales = category_demand.groupby(['Location', 'Product'])['Demand'].apply(
            lambda x: x.rolling(window=7, min_periods=1).mean()).reset_index(drop=True)

        category_demand.loc[:, 'Safety_Stock'] = rolling_avg_sales.values * safety_days
        demand_data.loc[category_demand.index, 'Safety_Stock'] = category_demand['Safety_Stock']

    return demand_data

# Calculate order fulfillment
def calculate_fulfillment(stock_data, demand_data, reservation_ratios):
    daily_results = []

    for day in range(1, num_days + 1):
        day_demand = demand_data[demand_data['Day'] == day]
        day_stock = stock_data.copy()
        day_stock['Available_Stock'] = day_stock['Initial_Stock']

        for product in products:
            category = category_map[product]
            ratio = reservation_ratios[category]

            for location in stores:
                location_demand = day_demand[(day_demand['Location'] == location) & (day_demand['Product'] == product)]
                safety_stock = location_demand['Safety_Stock'].iloc[0] if not location_demand.empty else 0
                avl_stock = day_stock.loc[(day_stock['Location'] == location) & (day_stock['Product'] == product), 'Available_Stock'].iloc[0]
                shared_stock = max(0, (avl_stock - safety_stock) * ratio)

                if location == "Online Platform":
                    online_fulfillment = min(location_demand['Demand'].sum(), avl_stock)
                    day_stock.loc[(day_stock['Location'] == location) & (day_stock['Product'] == product), 'Available_Stock'] -= online_fulfillment
                else:
                    offline_fulfillment = min(location_demand['Demand'].sum(), avl_stock - shared_stock)
                    day_stock.loc[(day_stock['Location'] == location) & (day_stock['Product'] == product), 'Available_Stock'] -= offline_fulfillment

                daily_results.append({
                    'Day': day,
                    'Location': location,
                    'Product': product,
                    'Stock_Level': avl_stock,
                    'Shared_Stock': shared_stock,
                    'Fulfillment': offline_fulfillment if location != "Online Platform" else online_fulfillment,
                    'Demand': location_demand['Demand'].sum()
                })

    return pd.DataFrame(daily_results)

# Perform calculations
demand_data = calculate_safety_stock(demand_data, category_safety_days)
fulfillment_data = calculate_fulfillment(stock_data, demand_data, category_reservation_ratio)

# Calculate and plot fulfillment rate
fulfillment_rate = fulfillment_data.groupby(['Location', 'Product']).sum().reset_index()
fulfillment_rate['Fulfillment_Rate'] = fulfillment_rate['Fulfillment'] / fulfillment_rate['Demand']

# Create a pivot table for the heatmap
fulfillment_pivot = fulfillment_rate.pivot_table(index='Location', columns='Product', values='Fulfillment_Rate')

fig = px.imshow(fulfillment_pivot, 
                labels={'x': 'Product', 'y': 'Location', 'color': 'Fulfillment Rate'},
                x=fulfillment_pivot.columns, 
                y=fulfillment_pivot.index,
                aspect='auto',
                title='Fulfillment Rate Heatmap')

# Visualize Data
st.plotly_chart(fig)

# Detailed view for selected components
st.header("Detailed Analysis")
selected_detail_stores = st.multiselect("Select Stores for Detailed Analysis", stores, default=stores, key='store_selector_2')
selected_detail_products = st.multiselect("Select Products for Detailed Analysis", products, default=products, key='product_selector_2')

if selected_detail_stores and selected_detail_products:
    detail_data = fulfillment_data[
        (fulfillment_data['Location'].isin(selected_detail_stores)) &
        (fulfillment_data['Product'].isin(selected_detail_products))
    ]

    # Aggregate data by day for selected stores/products
    detail_summary = detail_data.groupby('Day').agg({
        'Stock_Level': 'sum',
        'Demand': 'sum',
        'Shared_Stock': 'sum',
        'Fulfillment': 'sum'
    }).reset_index()

    detail_summary['Fulfillment_Rate'] = detail_summary['Fulfillment'] / detail_summary['Demand']

    # Plot line chart for detailed analysis
    # line_fig = px.line(detail_summary, x='Day', y=['Stock_Level', 'Demand', 'Shared_Stock', 'Fulfillment_Rate'],
    #                    labels={'value': 'Level', 'variable': 'Metric'},
    #                    title='Detailed Analysis Over Time',
    #                    markers=True)

    line_fig = px.line(detail_summary, x='Day', y=['Demand', 'Shared_Stock'],
                       labels={'value': 'Level', 'variable': 'Metric'},
                       title='Detailed Analysis Over Time',
                       markers=True)
    
    st.plotly_chart(line_fig)
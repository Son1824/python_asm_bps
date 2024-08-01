import pandas as pd
import matplotlib.pyplot as plt

# Read data from CSV files
customer_df = pd.read_csv('CustomerTable.csv')
product_df = pd.read_csv('ProductTable.csv')
transactions_df = pd.read_csv('TransactionsTable.csv')
product_category_df = pd.read_csv('ProductCategoryTable.csv')
market_trend_df = pd.read_csv('MarketTrendTable.csv')
website_access_category_df = pd.read_csv('WebsiteAccessCategoryTable.csv')

# Bar Chart: Number of Products by Product Category
print("Transactions Data Types:", transactions_df.dtypes)
product_category_count = product_df.groupby('CategoryID')['ProductID'].count()

# Create a bar chart for products by category
plt.figure(figsize=(10, 6))
product_category_count.plot(kind='bar')
plt.title('Number of Products by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Number of Products')
plt.xticks(ticks=range(len(product_category_count.index)),
           labels=product_category_df['CategoryName'].values, rotation=45)
plt.grid(axis='y')
plt.show()

# Line Chart: Average of EconomicIndicator by Year and Quarter
market_trend_df['Date'] = pd.to_datetime(market_trend_df['Date'])
market_trend_df['Year'] = market_trend_df['Date'].dt.year
market_trend_df['Quarter'] = market_trend_df['Date'].dt.quarter

# Create a line chart
plt.figure(figsize=(12, 6))
plt.plot(market_trend_df['Date'], market_trend_df['EconomicIndicator'], marker='o')
plt.title('Average of EconomicIndicator by Year, and Quarter')
plt.xlabel('Date')
plt.ylabel('Average of EconomicIndicator')
plt.grid()

# Customize x-ticks to show Year, and Quarter
step = len(market_trend_df) // 10
labels = [
    f"{row['Year']} Q{row['Quarter']}" if i % step == 0 else ''
    for i, row in market_trend_df.iterrows()
]
plt.xticks(ticks=market_trend_df['Date'][::step], labels=labels[::step], rotation=45, ha='right')
plt.show()

# Pie Chart: Count of CategoryID by PromotionalActivity
promotional_activity_count = market_trend_df['PromotionalActivity'].value_counts()

plt.figure(figsize=(10, 6))
promotional_activity_count.plot(kind='pie', autopct='%1.1f%%', startangle=140)
plt.title('Count of CategoryID by PromotionalActivity')
plt.ylabel('')
plt.show()
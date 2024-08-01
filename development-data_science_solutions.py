import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load data from CSV files
customer_df = pd.read_csv('CustomerTable.csv')
product_df = pd.read_csv('ProductTable.csv')
transactions_df = pd.read_csv('TransactionsTable.csv')
product_category_df = pd.read_csv('ProductCategoryTable.csv')
market_trend_df = pd.read_csv('MarketTrendTable.csv')
website_access_category_df = pd.read_csv('WebsiteAccessCategoryTable.csv')

# Display data types of transactions_df
print("Transactions Data Types:", transactions_df.dtypes)

# Create a revenue column in the Transactions dataframe
transactions_df['Revenue'] = transactions_df['Quantity'] * transactions_df['Price']

# Convert the date column to datetime format
transactions_df['Date'] = pd.to_datetime(transactions_df['Date'])

# Aggregate data to daily revenue
daily_revenue = transactions_df.groupby('Date')['Revenue'].sum().reset_index()

# Generate additional time-based features (e.g., day, month, year)
daily_revenue['Day'] = daily_revenue['Date'].dt.day
daily_revenue['Month'] = daily_revenue['Date'].dt.month
daily_revenue['Year'] = daily_revenue['Date'].dt.year

# Use only relevant features for the model
features = ['Day', 'Month', 'Year']
target = 'Revenue'

# Extract features and target variable
X = daily_revenue[features]
y = daily_revenue[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Plot actual vs predicted revenue
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Revenue')
plt.plot(y_pred, label='Predicted Revenue')
plt.title('Actual vs Predicted Revenue')
plt.xlabel('Date Index')
plt.ylabel('Revenue')
plt.legend()
plt.show()

# Create a dataframe for future dates
future_dates = pd.date_range(start='2023-08-01', end='2023-12-31', freq='D')
future_df = pd.DataFrame({'Date': future_dates})

# Generate time-based features for future dates
future_df['Day'] = future_df['Date'].dt.day
future_df['Month'] = future_df['Date'].dt.month
future_df['Year'] = future_df['Date'].dt.year

# Predict future revenue
future_features = future_df[features]
future_df['Predicted_Revenue'] = model.predict(future_features)

# Plot future revenue predictions
plt.figure(figsize=(12, 6))
plt.plot(future_df['Date'], future_df['Predicted_Revenue'], marker='o')
plt.title('Predicted Future Sales')
plt.xlabel('Date')
plt.ylabel('Predicted Revenue')
plt.grid()
plt.show()

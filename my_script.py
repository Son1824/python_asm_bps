import pandas as pd

# Load the dataset into a DataFrame
df = pd.read_csv("CustomerTable.csv")

# Fill missing value in 'Name' column with 'Unknown'
df['Name'] = df['Name'].fillna('Unknown')

# Example: Remove rows with invalid email addresses
df = df[df['Email'].str.contains('@', na=False)]

# Print the DataFrame to check the result
print(df)
df.to_csv("CustomerTable.csv", index=False)
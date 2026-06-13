import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("house.csv")

# Split data
X = df.drop('medv', axis=1)
y = df['medv']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# UI
st.title("House Price Prediction")

# Input fields (simplified)
rm = st.number_input("Average number of rooms", value=6.0)
lstat = st.number_input("Lower status population (%)", value=10.0)
ptratio = st.number_input("Pupil-teacher ratio", value=18.0)

# Predict button
if st.button("Predict Price"):
    # Create input data (fill others with average)
    input_data = X.mean().values.reshape(1, -1).copy()
    
    # Replace selected features
    input_data[0][X.columns.get_loc('rm')] = rm
    input_data[0][X.columns.get_loc('lstat')] = lstat
    input_data[0][X.columns.get_loc('ptratio')] = ptratio

    prediction = model.predict(input_data)

    # Convert to INR
    price_usd = prediction[0] * 1000
    price_inr = price_usd * 83

    st.success(f"Predicted Price: ₹{price_inr:,.2f}")
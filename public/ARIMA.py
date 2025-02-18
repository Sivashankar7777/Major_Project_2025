#ARIMA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Function to load dataset
def load_data(file_path):
    try:
        # Load the dataset without parsing dates initially
        data = pd.read_csv(file_path)

        # Print column names for debugging
        print("Column names in the dataset:", data.columns.tolist())

        # Automatically identify the date column
        date_col = None
        for col in data.columns:
            # Try converting to datetime to check if the column is a date
            try:
                pd.to_datetime(data[col])
                date_col = col
                break  # Exit the loop if a date column is found
            except (ValueError, TypeError):
                continue

        if date_col is None:
            raise ValueError("No date column found in the dataset.")

        # Parse the identified date column and set as index
        data['Date'] = pd.to_datetime(data[date_col])  # Convert identified column to datetime
        data.set_index('Date', inplace=True)  # Set 'Date' as the index
        print(f"Using '{date_col}' as the date column.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Function to plot the time series
def plot_time_series(series):
    plt.figure(figsize=(10, 5))
    plt.plot(series)
    plt.title('Time Series Data')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid()
    plt.show()

# Function to fit ARIMA model and forecast
def fit_arima(series, order=(1, 1, 1), steps=10):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    print(model_fit.summary())

    # Forecast future values
    forecast = model_fit.forecast(steps=steps)
    return forecast

# Main function to run the workflow
def main():
    # Get the dataset from the user
    file_path = ('stock.csv')

    # Load the dataset
    data = load_data(file_path)
    if data is None:
        return

    # Get the first value column (assumed to be the second column)
    value_col = data.columns[1]  # You can modify this logic if needed

    # Check if the value column exists
    if value_col not in data.columns:
        print(f"Error: '{value_col}' column not found in the dataset.")
        return

    series = data[value_col]  # Use the first value column

    # Plot the time series
    plot_time_series(series)

    # Plot ACF and PACF
    plot_acf(series)
    plt.title('ACF Plot')
    plt.show()

    plot_pacf(series)
    plt.title('PACF Plot')
    plt.show()

    # Fit ARIMA model
    forecast = fit_arima(series)

    # Print forecasted values
    print('Forecasted Values:')
    print(forecast)

    # Plot the forecast
    plt.figure(figsize=(10, 5))
    plt.plot(series, label='Historical Data')
    plt.plot(pd.date_range(series.index[-1], periods=11, freq='M')[1:], forecast, label='Forecasted Values', color='red')
    plt.title('Forecasted Values vs Historical Data')
    plt.xlabel('Date')
    plt.ylabel(value_col)
    plt.legend()
    plt.grid()
    plt.show()

# Run the main function
if __name__ == "__main__":
    main()
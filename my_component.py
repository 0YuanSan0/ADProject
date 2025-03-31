import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller

"""
Perform an Augmented Dickey-Fuller (ADF) test on a time series DataFrame.
This test helps determine if a time series is stationary.

Args:
    df_timeseries (pd.DataFrame): A DataFrame containing time series data with a 'Price' column.

Returns:
    None
"""
def adf_check(df_timeseries):
    # Extract the 'Price' column from the DataFrame
    prices = df_timeseries.Price
    # Perform the ADF test
    check_result = adfuller(prices)
    # Print the ADF score
    print("ADF Score: ", check_result[0])
    # Print the p-value
    print("p-value: ", check_result[1])
    # Print the critical values
    for key, value in check_result[4].items():
        print(f"\t{key}: {value}")


"""
Resample the time series data on a monthly basis.
This is useful for aggregating daily data into monthly data for comparison.

Args:
    df_timeseries (pd.DataFrame): A DataFrame containing time series data with a datetime index.

Returns:
    pd.DataFrame: A new DataFrame with monthly resampled data.
"""
def month_resample(df_timeseries):
    # Resample the data on a monthly basis and calculate the mean
    df_timeseries_monthly = df_timeseries.resample('ME').mean()
    return df_timeseries_monthly


"""
Read a CSV file, perform basic data processing, and return a time series DataFrame.
This function also performs an ADF test and Min-Max scaling on the data.

Args:
    df_path (str): The path to the CSV file.

Returns:
    tuple: A tuple containing the processed time series DataFrame and the MinMaxScaler object.
"""
def process_data(df_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(df_path)
    # print("DataFrame Info:")
    # print(df.info())
    # print("\nDataFrame Description:")
    # print(df.describe())

    df[['Price', 'High', 'Low']] = df[['Price', 'High', 'Low']].astype(float)
    df['Price'] = df[['Price', 'High', 'Low']].mean(axis=1) # Calculate typical price

    df_timeseries = df[['Date', 'Price']].copy() # Only reserved dates + typical prices
    df_timeseries['Date'] = pd.to_datetime(df_timeseries['Date']) # Normalized date format
    df_timeseries.sort_values(by='Date', ascending=True, inplace=True)
    df_timeseries.reset_index(drop=True, inplace=True)
    df_timeseries.set_index('Date', inplace=True) # Set the date as an index
    adf_check(df_timeseries) # The ADF test showed no stationarity
    print("Data overview: ")
    show_line_chart(df_timeseries)

    # Min-Max scaling
    price = df_timeseries.Price.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler.fit(price)
    df_timeseries['Price'] = scaler.transform(price)
    df_timeseries['Price'] = df_timeseries['Price'].astype('float64')

    print("DataFrame Info:")
    print(df_timeseries.info())
    print("\nDataFrame Description:")
    print(df_timeseries.describe())

    # Save the transformed data to a CSV file
    df_timeseries.to_csv('./Dataset/SilverPrices_transformed.csv')

    return df_timeseries, scaler


"""
Divide the time series DataFrame into training and test sets.
The training set contains data from 2013 to 2021, and the test set contains data from 2022 to 2023.

Args:
    sliver_df (pd.DataFrame): A DataFrame containing time series data with a datetime index.

Returns:
    tuple: A tuple containing the training and test DataFrames.
"""
def train_test_divide(sliver_df):
    # print(sliver_df.columns)
    # Select data from 2013 to 2021 for the training set
    train_df = sliver_df[sliver_df.index.year < 2022]
    # Select data from 2022 to 2023 for the test set
    test_df = sliver_df[sliver_df.index.year >= 2022]
    return train_df, test_df


"""
Show a line chart of the time series data.

Args:
    df (pd.DataFrame): A DataFrame containing time series data with a datetime index and a 'Price' column.

Returns:
    None
"""
def show_line_chart(df):
    plt.figure(figsize=(15, 6), dpi=150)
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rc('axes', edgecolor='black')
    plt.plot(df.index, df.Price, color='black', lw=1)

    # Set the title of the chart
    plt.title('Sliver Price Training and Test Sets', fontsize=14)
    plt.xlabel('Date', fontsize=10)
    plt.ylabel('Price', fontsize=10)
    plt.legend(['Silver Price'], loc='upper left', prop={'size': 14})
    plt.grid(color='gray')
    plt.show()


"""
Evaluate the performance of a prediction model.
This function calculates evaluation metrics and visualizes the results.

Args:
    train_df (pd.DataFrame): The training DataFrame.
    test_df (pd.DataFrame): The test DataFrame.
    predict_price (np.ndarray): The predicted prices.
    scaler (MinMaxScaler): The MinMaxScaler object used for scaling.

Returns:
    None
"""
def evaluation_model(train_df, test_df, predict_price, scaler):
    # Inverse Min-Max: shrinks data from 0 to 1 back to the original size
    train_df.loc[:, "Price"] = scaler.inverse_transform(train_df.Price.values.reshape(-1, 1)).flatten()
    test_df.loc[:, "Price"] = scaler.inverse_transform(test_df.Price.values.reshape(-1, 1)).flatten()
    predict_price = scaler.inverse_transform(predict_price.reshape(-1, 1)).flatten()

    # Fill the empty prediction window with truth values
    if len(test_df) > len(predict_price):
        predict_price = np.concatenate((test_df[:-len(predict_price)].Price.values,
                                    predict_price), axis=0)

    error_metrics(test_df.Price.values, predict_price) # Score evaluation
    show_line_chart_predict(train_df, test_df, predict_price) # Visual evaluation


"""
Show a line chart comparing the training data, true values, and predicted values.

Args:
    train_df (pd.DataFrame): The training DataFrame.
    test_df (pd.DataFrame): The test DataFrame.
    predict_price (np.ndarray): The predicted prices.

Returns:
    None
"""
def show_line_chart_predict(train_df, test_df, predict_price):
    plt.figure(figsize=(15, 6), dpi=150)
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rc('axes', edgecolor='black')
    plt.plot(train_df.index, train_df.Price, color='black', lw=1)
    plt.plot(test_df.index, test_df.Price, color='blue', lw=1)
    plt.plot(test_df.index, predict_price, color='magenta', lw=1)

    plt.title('Sliver Price True and Predict Values', fontsize=14)
    plt.xlabel('Date', fontsize=10)
    plt.ylabel('Price', fontsize=10)
    plt.legend(['Train values', 'True values', 'Predict values'], loc='upper left', prop={'size': 14})
    plt.grid(color='gray')
    plt.show()


"""
Calculate evaluation metrics (MSE, RMSE, MAE, MAPE) for a prediction model.

Args:
    y_true (np.ndarray): The true values.
    y_pred (np.ndarray): The predicted values.

Returns:
    tuple: A tuple containing the MSE, RMSE, MAE, and MAPE values.
"""
def error_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    # If there are zero values in y_true, calculate MAPE after filtering out the zero values
    non_zero_indices = y_true != 0
    if np.any(non_zero_indices):
        mape = 100 * mean_absolute_percentage_error(y_true[non_zero_indices], y_pred[non_zero_indices])
    else:
        mape = np.nan
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"MAPE: {mape}%")
    return mse, rmse, mae, mape

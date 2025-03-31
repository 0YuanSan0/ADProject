import numpy as np
from sklearn.ensemble import RandomForestRegressor

"""
Prepare the data for a Random Forest regressor.
This function divides the time series data into input features (X) and target values (Y).

Args:
    df (pd.DataFrame): A DataFrame containing time series data with a 'Price' column.
    window_size (int): The size of the sliding window used to create the input features.

Returns:
    tuple: A tuple containing the input features (X) and target values (Y) as NumPy arrays.
"""
def process_df_for_rf(df, window_size: int):
    # Reshape the 'Price' column to a 2D array
    data = df.Price.values.reshape(-1, 1)
    x_data = []
    y_data = []

    # Create the input features and target values using a sliding window
    for i in range(window_size, len(data)):
        x_data.append(data[i - window_size:i, 0])
        y_data.append(data[i, 0])
    x_data = np.array(x_data)
    y_data = np.array(y_data).flatten()

    return x_data, y_data


"""
Make predictions using a Random Forest regressor.

Args:
    train_df (pd.DataFrame): The training DataFrame.
    test_df (pd.DataFrame): The test DataFrame.
    window_size (int): The size of the sliding window used to create the input features.
    n_estimators (int): The number of trees in the Random Forest regressor.

Returns:
    np.ndarray: The predicted values.
"""
def rf_predict(train_df, test_df, window_size=60,n_estimators=100):
    # Prepare the training data
    x_train, y_train = process_df_for_rf(train_df, window_size)
    # Create a Random Forest regressor
    regressor = RandomForestRegressor(n_estimators)
    regressor.fit(x_train, y_train)

    # Prepare the test data
    x_test, y_test = process_df_for_rf(test_df, window_size)
    # Make predictions on the test data
    y_pred = regressor.predict(x_test)

    return y_pred

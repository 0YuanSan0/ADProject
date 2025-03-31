from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

"""
    Perform time series forecasting using the SARIMAX model.

    Args:
        train_df (pandas.DataFrame): The training DataFrame containing the historical data.
        test_df (pandas.DataFrame): The testing DataFrame for which predictions are to be made.

    Returns:
        numpy.ndarray: An array containing the predicted values.
"""
def sarimax_predict(train_df, test_df):
    # Calculate the number of steps to predict, which is equal to the length of the testing DataFrame
    predict_steps = len(test_df)
    # Initialize the SARIMAX model with specified parameters
    model = SARIMAX(
        train_df.Price,
        order=(3, 0, 2),  # (p,d,q)
        seasonal_order=(3, 1, 2, 24),  # (S,P,D,Q,S)
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit()
    results.summary()

    # Predict the future
    forecast = results.get_forecast(steps=predict_steps)
    # Get the mean of the predictions
    y_pred  = np.array(forecast.predicted_mean).astype('float64')

    return y_pred

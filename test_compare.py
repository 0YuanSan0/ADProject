from my_lstm import lstm_model_predict
from my_random_forest import rf_predict
from my_sarimax import sarimax_predict
import my_component
from my_component import evaluation_model

"""
Compare the performance of different prediction models (SARIMAX, LSTM, Random Forest).

Args:
    df (pd.DataFrame): A DataFrame containing time series data.
    scaler (MinMaxScaler): The MinMaxScaler object used for scaling.

Returns:
    None
"""
def test_compare(df, scaler):
    # Divide the data into training and test sets
    train_df, test_df = my_component.train_test_divide(df)

    # Test the SARIMAX model
    print("SARIMX model test: ")
    predict = sarimax_predict(train_df.copy(), test_df.copy())
    evaluation_model(train_df.copy(), test_df.copy(), predict, scaler)

    # Test the LSTM model
    print("LSTM model test: ")
    predict = lstm_model_predict(train_df.copy(), test_df.copy(), "Config/lstm_basetest_config.json")
    evaluation_model(train_df.copy(), test_df.copy(), predict, scaler)

    # Test the Random Forest model
    print("Random forest model test: ")
    predict = rf_predict(train_df.copy(), test_df.copy(), 5, 50)
    evaluation_model(train_df.copy(), test_df.copy(), predict, scaler)


if __name__ == "__main__":
    sliver_df, scaler = my_component.process_data("./Dataset/SilverPrices.csv")
    sliver_df = my_component.month_resample(sliver_df)  # Resampling was performed monthly during testing
    test_compare(sliver_df, scaler)


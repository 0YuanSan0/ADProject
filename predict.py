from my_lstm import lstm_model_predict
from my_random_forest import rf_predict
import my_component
from my_component import evaluation_model

"""
    Perform predictions using LSTM and Random Forest models.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.
        scaler: The scaler object used for data scaling.
"""
def predict(df, scaler):
    train_df, test_df = my_component.train_test_divide(df)

    # Perform LSTM model prediction
    print("LSTM model predicts:")
    lstm_predict = lstm_model_predict(train_df.copy(), test_df.copy(), "Config/lstm_config.json")
    evaluation_model(train_df.copy(), test_df.copy(), lstm_predict, scaler)

    # Perform Random Forest model prediction
    print("Random forest model predicts:")
    _rf_predict = rf_predict(train_df.copy(), test_df.copy(), 60, 100)
    evaluation_model(train_df.copy(), test_df.copy(), _rf_predict, scaler)

if __name__ == "__main__":
    sliver_df, scaler = my_component.process_data("./Dataset/SilverPrices.csv")
    predict(sliver_df, scaler)
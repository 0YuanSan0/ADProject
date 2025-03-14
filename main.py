import pandas as pd

from my_lstm import lstm_model_predict
from my_random_forest import rf_predict
from my_sarimax import sarimax_predict
import my_component
from my_component import evaluation_model


def test_compare(df, scaler):
    sliver_df = my_component.month_resample(df)  # 按月重新采样
    train_df, test_df = my_component.train_test_divide(sliver_df) # 分割数据集
    print("SARIMX模型测试：")
    predict = sarimax_predict(train_df.copy(), test_df.copy())
    evaluation_model(train_df.copy(), test_df.copy(), predict, scaler)

    print("LSTM模型测试：")
    predict = lstm_model_predict(train_df.copy(), test_df.copy(), "Config/lstm_basetest_config.json")
    evaluation_model(train_df.copy(), test_df.copy(), predict, scaler)

    print("随机森林模型测试：")
    predict = rf_predict(train_df.copy(), test_df.copy(), 5, 50)
    evaluation_model(train_df.copy(), test_df.copy(), predict, scaler)


def predict(df, scaler):
    train_df, test_df = my_component.train_test_divide(df)
    print("LSTM模型推理：")
    lstm_predict = lstm_model_predict(train_df.copy(), test_df.copy(), "Config/lstm_config.json")
    evaluation_model(train_df.copy(), test_df.copy(), lstm_predict, scaler)
    print("随机森林模型推理：")
    _rf_predict = rf_predict(train_df.copy(), test_df.copy(), 60, 100)
    evaluation_model(train_df.copy(), test_df.copy(), _rf_predict, scaler)


if __name__ == "__main__":
    sliver_df = pd.read_csv("./Dataset/SilverPrices.csv")
    sliver_df, scaler = my_component.process_data(sliver_df)
    # test_compare(sliver_df, scaler)
    predict(sliver_df, scaler)

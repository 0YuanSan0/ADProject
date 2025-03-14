from my_lstm import lstm_model_predict
from my_random_forest import rf_predict
from my_sarimax import sarimax_predict
import my_component
from my_component import evaluation_model


def test_compare(df, scaler):
    train_df, test_df = my_component.train_test_divide(df) # 分割数据集
    print("SARIMX model test: ") # SARIMX模型测试：
    predict = sarimax_predict(train_df.copy(), test_df.copy())
    evaluation_model(train_df.copy(), test_df.copy(), predict, scaler)

    print("LSTM model test: ") # LSTM模型测试：
    predict = lstm_model_predict(train_df.copy(), test_df.copy(), "Config/lstm_basetest_config.json")
    evaluation_model(train_df.copy(), test_df.copy(), predict, scaler)

    print("Random forest model test: ") # 随机森林模型测试：
    predict = rf_predict(train_df.copy(), test_df.copy(), 5, 50)
    evaluation_model(train_df.copy(), test_df.copy(), predict, scaler)


if __name__ == "__main__":
    sliver_df, scaler = my_component.process_data("./Dataset/SilverPrices.csv")
    sliver_df = my_component.month_resample(sliver_df)  # Resampling was performed monthly during testing | 测试时按月重新采样
    test_compare(sliver_df, scaler)


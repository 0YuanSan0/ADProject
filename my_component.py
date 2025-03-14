import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller

"""
ADF Score
ADF分数
"""
def adf_check(df_timeseries):
    prices = df_timeseries.Price
    check_result = adfuller(prices)
    print("ADF Score：", check_result[0])
    print("p-value：", check_result[1])
    for key, value in check_result[4].items():
        print(f"\t{key}: {value}")



"""
Resampling monthly for comparison
按月重新采样，用于对比
"""
def month_resample(df_timeseries):
    df_timeseries_monthly = df_timeseries.resample('ME').mean()
    return df_timeseries_monthly


"""
Read database, simple processing
读取数据库，简单处理
"""
def process_data(df_path):
    df = pd.read_csv(df_path)
    df[['Price', 'High', 'Low']] = df[['Price', 'High', 'Low']].astype(float)
    df['Price'] = df[['Price', 'High', 'Low']].mean(axis=1) # Calculate typical price | 计算典型价格

    df_timeseries = df[['Date', 'Price']].copy() # Only reserved dates + typical prices | 只保留日期 + 典型价格
    df_timeseries['Date'] = pd.to_datetime(df_timeseries['Date']) # Normalized date format | 规范化日期格式
    df_timeseries.sort_values(by='Date', ascending=True, inplace=True)
    df_timeseries.reset_index(drop=True, inplace=True)
    df_timeseries.set_index('Date', inplace=True) # Set the date as an index | 将日期设置为索引
    adf_check(df_timeseries) # The ADF test showed no stationarity | ADF检测，结果是没有平稳性
    print("Data overview: ") # 数据总览：
    show_line_chart(df_timeseries)

    price = df_timeseries.Price.values.reshape(-1, 1)
    scaler = MinMaxScaler() # Min-Max归一化，加速计算
    scaler.fit(price)
    df_timeseries['Price'] = scaler.transform(price)
    df_timeseries['Price'] = df_timeseries['Price'].astype('float64')
    return df_timeseries, scaler


"""
Split the training set and test set (training from 2013 to 2021, testing from 2022 to 2023)
分割训练集和测试集（13~21年训练，22~23年测试）
"""
def train_test_divide(sliver_df):
    # print(sliver_df.columns)
    train_df = sliver_df[sliver_df.index.year < 2022]
    test_df = sliver_df[sliver_df.index.year >= 2022]
    return train_df, test_df


"""
Overview drawing
总览绘图
"""
def show_line_chart(df):
    plt.figure(figsize=(15, 6), dpi=150)
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rc('axes', edgecolor='black')
    plt.plot(df.index, df.Price, color='black', lw=1)

    plt.title('Sliver Price Training and Test Sets', fontsize=14)
    plt.xlabel('Date', fontsize=10)
    plt.ylabel('Price', fontsize=10)
    plt.legend(['Silver Price'], loc='upper left', prop={'size': 14})
    plt.grid(color='gray')
    plt.show()


"""
Evaluation result
评估结果
"""
def evaluation_model(train_df, test_df, predict_price, scaler):
    # Inverse Min-Max: shrinks data from 0 to 1 back to the original size | 反向Min-Max，将数据从0~1放缩回原来的大小
    train_df.loc[:, "Price"] = scaler.inverse_transform(train_df.Price.values.reshape(-1, 1)).flatten()
    test_df.loc[:, "Price"] = scaler.inverse_transform(test_df.Price.values.reshape(-1, 1)).flatten()
    predict_price = scaler.inverse_transform(predict_price.reshape(-1, 1)).flatten()

    # Fill the empty prediction window with truth values | 用真值填充预测空窗口
    if len(test_df) > len(predict_price):
        predict_price = np.concatenate((test_df[:-len(predict_price)].Price.values,
                                    predict_price), axis=0)


    error_metrics(test_df.Price.values, predict_price) # Score evaluation | 分数评估
    show_line_chart_predict(train_df, test_df, predict_price) # Visual evaluation | 可视化评估


"""
Evaluation drawing
评估绘图
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
Evaluation scores MSE, RMSE, MAE, MAPE
评估分数 MSE, RMSE, MAE, MAPE
"""
def error_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    # If there are zero values in y_true, calculate MAPE after filtering out the zero values
    # 处理 y_true 中存在零值的情况，过滤掉零值后计算 MAPE
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

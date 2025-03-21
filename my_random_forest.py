import numpy as np
from sklearn.ensemble import RandomForestRegressor

"""
Self Train
Divide X, Y
Use X: [i, i + window_size-1]
Prediction Y: [i + window_size]
Difference between LSTM and LSTM: The shape of the final X, Y is not the same as the shape of the final X, Y of LSTM

自己训练自己
分割X, Y
用   X: [i, i + window_size -1]
预测 Y: [i + window_size]
和LSTM的区别：最后X, Y的形状和LSTM最后X, Y的形状不一样
"""
def process_df_for_rf(df, window_size: int):
    data = df.Price.values.reshape(-1, 1)
    x_data = []
    y_data = []

    for i in range(window_size, len(data)):
        x_data.append(data[i - window_size:i, 0])
        y_data.append(data[i, 0])

    x_data = np.array(x_data)
    y_data = np.array(y_data).flatten()

    return x_data, y_data


def rf_predict(train_df, test_df, window_size=60,n_estimators=100):
    x_train, y_train = process_df_for_rf(train_df, window_size)
    regressor = RandomForestRegressor(n_estimators)
    regressor.fit(x_train, y_train)

    x_test, y_test = process_df_for_rf(test_df, window_size)
    y_pred = regressor.predict(x_test)

    return y_pred

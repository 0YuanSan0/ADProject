from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

def sarimax_predict(train_df, test_df):
    predict_steps = len(test_df)
    model = SARIMAX(
        train_df.Price,
        order=(3, 0, 2),  # (p,d,q)
        seasonal_order=(3, 1, 2, 24),  # (S,P,D,Q,S)
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit()
    results.summary()

    # 预测未来
    forecast = results.get_forecast(steps=predict_steps)
    # 获取预测的均值
    y_pred  = np.array(forecast.predicted_mean).astype('float64')
    return y_pred

import json
import os.path

import numpy as np
from keras import Model
from keras.api.layers import Input, Dropout, Dense, LSTM
from keras.api.models import load_model


def process_df_for_lstm(df, window_size:int):
    data = df.Price.values.reshape(-1, 1)
    x_data = []
    y_data = []

    for i in range(window_size, len(data)):
        x_data.append(data[i - window_size:i, 0])
        y_data.append(data[i, 0])
    x_data = np.array(x_data)
    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))
    y_data = np.array(y_data)
    y_data = np.reshape(y_data, (-1, 1))
    return x_data, y_data


def lstm_model(config):
    input1 = Input(shape=(config["window_size"], 1))
    x = LSTM(units=64, return_sequences=True)(input1)
    x = Dropout(config["drop_rate"])(x)
    x = LSTM(units=64, return_sequences=True)(x)
    x = Dropout(config["drop_rate"])(x)
    x = LSTM(units=64)(x)
    x = Dropout(config["drop_rate"])(x)
    x = Dense(32, activation=config["activation"])(x)
    dnn_output = Dense(1)(x)

    model = Model(inputs=input1, outputs=[dnn_output])
    model.compile(loss=config["loss"], optimizer=config["optimizer"])
    model.summary()

    return model


def lstm_model_predict(train_df, test_df,config_path):

    with open(config_path, encoding="UTF-8") as file:
        config = json.load(file)

    model_path = config["model_path"]

    if os.path.exists(model_path):
        print("加载已训练模型，直接进行推理")
        model = load_model(model_path)
        # model.compile(loss=config["loss"], optimizer=config["optimizer"])
    else:
        print("无已训练模型，开始训练")
        x_train, y_train = process_df_for_lstm(train_df, config["window_size"])
        model = lstm_model(config)
        model.fit(x_train, y_train, epochs=config["epochs"], batch_size=config["batch_size"],
                  validation_split=config["validation_split"], verbose=1)
        print("训练完成，保存模型")
        model.save(model_path)

    print("开始推理")
    x_test, y_test = process_df_for_lstm(test_df, config["window_size"])
    model.evaluate(x_test, y_test)
    y_pred = model.predict(x_test)

    return y_pred

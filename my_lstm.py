import json
import os.path

import numpy as np
from keras import Model
from keras.api.layers import Input, Dropout, Dense, LSTM
from keras.api.models import load_model

"""
Prepare the data for an LSTM model.
This function divides the time series data into input features (X) and target values (Y).

Args:
    df (pd.DataFrame): A DataFrame containing time series data with a 'Price' column.
    window_size (int): The size of the sliding window used to create the input features.

Returns:
    tuple: A tuple containing the input features (X) and target values (Y) as NumPy arrays.
"""
def process_df_for_lstm(df, window_size: int):
    # Reshape the 'Price' column to a 2D array
    data = df.Price.values.reshape(-1, 1)
    x_data = []
    y_data = []

    # Create the input features and target values using a sliding window
    for i in range(window_size, len(data)):
        x_data.append(data[i - window_size:i, 0])
        y_data.append(data[i, 0])
    x_data = np.array(x_data)
    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))
    y_data = np.array(y_data)
    y_data = np.reshape(y_data, (-1, 1))

    return x_data, y_data


"""
Define an LSTM model.

Args:
    config (dict): A dictionary containing the model configuration.

Returns:
    Model: The compiled LSTM model.
"""
def lstm_model(config):
    # Define the input layer
    input1 = Input(shape=(config["window_size"], 1))
    # Add LSTM layers with dropout
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


"""
LSTM prediction (if has no trained model, will be trained first)
"""
def lstm_model_predict(train_df, test_df, config_path):
    with open(config_path, encoding="UTF-8") as file:
        config = json.load(file)

    model_path = config["model_path"]

    if os.path.exists(model_path):
        print("Load the trained model and make predictions directly") # Load the trained model and deduce directly
        model = load_model(model_path)
        # model.compile(loss=config["loss"], optimizer=config["optimizer"])
    else:
        print("No trained model, start training")
        x_train, y_train = process_df_for_lstm(train_df, config["window_size"])
        model = lstm_model(config)
        model.fit(x_train, y_train, epochs=config["epochs"], batch_size=config["batch_size"],
                  validation_split=config["validation_split"], verbose=1)
        print("Training complete, save the model")
        model.save(model_path)

    print("Start prediction")
    x_test, y_test = process_df_for_lstm(test_df, config["window_size"])
    model.evaluate(x_test, y_test)
    y_pred = model.predict(x_test)

    return y_pred

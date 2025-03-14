# Multi-model for silver forecasting project instructions

## Extract the project files and ensure that they are organized as follows：

```
root
  ├ Config
  │   ├ lstm_basetest_config.json
  │   └ lstm_config.json
  │
  ├ Dataset
  │	  └ SilverPrices.csv
  │   
  ├ lstm_model.h5 *
  ├ lstm_model_basetest.h5 *
  │
  ├ my_component.py
  ├ my_lstm.py
  ├ my_random_forest.py
  ├ my_sarimax.py
  ├ test_compare.py
  ├ predict.py
  │
  └ requirements.txt
```
+ **Config**:
  + lstm_basetest_config.json: Configuration of the LSTM model during comparison testing.
  + lstm_config.json: Configuration of the LSTM model during predictive testing.
+ **Data:**
  + SilverPrices.csv: Silver price data set from 2013 to 2023.
+ **weight (optional):**
  + lstm_model_basetest.h5 : LSTM is trained by **us** in comparison testing. (No model will start training again)
  + lstm_model.h5: The LSTM is trained by **us** in predictive testing. (No model will start training again)
+ **py:** python code file.
+ **requirements.txt:** Required python library


## Use

Ensure that the project documents are organized as above

1. Switch the working path to the root folder of the project file

   ```cmd
   cd /path_to_root/root
   ```

2. Install the required python libraries through requirements.txt

   ```cmd
   pip install -r requirements.txt
   ```

3. Test and compare

   ```cmd
   python test_compare.py
   ```
    example:
   ```pseudocode
   ADF Score： -1.998805019481718
   p-value： 0.2870973021664973
   	1%: -3.4329569285480814
   	5%: -2.862691729403106
   	10%: -2.5673831097880595
   	
   Data overview: 
   SARIMX model test: 
   
   * * *
   
   Machine precision = 2.220D-16
    N =           11     M =           10
   
   At X0         0 variables are exactly at the bounds
   
   At iterate    0    f= -1.96359D-02    |proj g|=  2.34566D-01
   
   At iterate    5    f= -1.97318D-02    |proj g|=  2.05848D+00
   
   ......
   
   At iterate   50    f= -1.00009D-01    |proj g|=  2.94713D+00
   
   * * *
   ......
   
   * * *
                
   
   MSE: 7.876935727101502
   RMSE: 2.806587915441364
   MAE: 2.03348268790561
   MAPE: 9.185571379236178%
   
   LSTM model test: 
   Load the trained model and make predictions directly
   WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
   Start prediction
   1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 269ms/step - loss: 0.0126
   1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 160ms/step
   MSE: 2.814691921732139
   RMSE: 1.6777043606464574
   MAE: 1.1928762556950705
   MAPE: 5.414655031603418%
   
   Random forest model test: 
   MSE: 1.8420709442734018
   RMSE: 1.35722914213975
   MAE: 0.9724879276180595
   MAPE: 4.509984320958819%
   ```
   
4. Finally Prediction

   ```cmd
   python predict.py
   ```
   example:
   ```pseudocode
   ADF Score： -1.998805019481718
   p-value： 0.2870973021664973
   	1%: -3.4329569285480814
   	5%: -2.862691729403106
   	10%: -2.5673831097880595
   Data overview: 
   
   LSTM model predicts:
   Load the trained model and make predictions directly
   Start prediction
   WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
   11/11 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - loss: 0.0012
   11/11 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step
   MSE: 0.23990032952399362
   RMSE: 0.4897962122393288
   MAE: 0.3603175473306691
   MAPE: 1.6213904800117331%
   
   Random forest model predicts:
   MSE: 0.1327424226438721
   RMSE: 0.3643383354025103
   MAE: 0.2672275408496724
   MAPE: 1.2133067794967076%
   ```


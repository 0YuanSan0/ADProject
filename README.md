# Multi-model for silver forecasting project instructions

## Extract the project files and ensure that they are organized as follows：

```
root
  ├ Config
  │   ├ lstm_basetest_config.json
  │	  └ lstm_config.json
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
  └
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
   ADF 分数： -1.998805019481718
   p-value： 0.2870973021664973
   	1%: -3.4329569285480814
   	5%: -2.862691729403106
   	10%: -2.5673831097880595
   数据总览：
   SARIMX模型测试：
   
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
   
   
   LSTM模型测试：
   加载已训练模型，直接进行推理
   开始推理
   WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
   1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 260ms/step - loss: 0.0126
   1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 154ms/step
   MSE: 2.814691921732139
   RMSE: 1.6777043606464574
   MAE: 1.1928762556950705
   MAPE: 5.414655031603418%
   
   随机森林模型测试：
   MSE: 1.9113858061448163
   RMSE: 1.3825287722665363
   MAE: 0.9933112261289953
   MAPE: 4.559875702334848%
   ```
   
4. Finally Prediction

   ```cmd
   python predict.py
   ```
   example:
   ```pseudocode
	ADF 分数： -1.998805019481718
   p-value： 0.2870973021664973
   	1%: -3.4329569285480814
   	5%: -2.862691729403106
   	10%: -2.5673831097880595
   数据总览：
   LSTM模型推理：
   加载已训练模型，直接进行推理
   WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
   开始推理
   11/11 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - loss: 0.0012
   11/11 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step
   MSE: 0.23990032952399362
   RMSE: 0.4897962122393288
   MAE: 0.3603175473306691
   MAPE: 1.6213904800117331%
   随机森林模型推理：
   MSE: 0.1398204970115734
   RMSE: 0.3739257907815044
   MAE: 0.27258595588235207
   MAPE: 1.2387041303433666%
   ```


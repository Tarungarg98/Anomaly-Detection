*** This is not a working code rather a dummy code due to sharing restrictions of data. You can use the framework and test on your data by making the required changes ***


What is it about ?

Offices, Industries and showrooms consumes a huge amount of power on a daily basis. Due to variation in the load requirements or in the power supply, there are huge variations in the required vs actual power consumption. Such fluctuations, may lead to power breakdown which may result in power cut or a short circuit leading to a huge power loss. In either case, it creates a huge impact to the fuctioning of the industry and can result in huge financial losses. In this project, we aim to detect any abnormal power flow and raise a alert in case of signifact variation in power consumption. 

How it works ?

The live and interactive user interface is designed using Dash-plotly. The parameters used to predict power consumption are passed as input parameters to the model. The model predicts the power consumption and returns a graph along with anomalies if any. Current faults, previous faults, and all faults displayed on the user interface. There is also an option to visualize the graphs along with input parameters to understand the cause of faults. 

How we made it ?

The model was created using a unsupervised learning technique. We were provided with a dataset containing temperature and dewpoint for indoors and outdoors, Day of the Week, Hour of Day, Meter Name, Store Number and Actual Power Consumption in KWh. First we started with the time series analysis of the data. The findings were that the data does not follow any significant seasonal or trend component. Thus it was decided to switch to Artificial Neural Networks as the time series did not gave significant results. For ANN, we encoded Day of Week and Hour of the Day using one-hot encoding as they don't have any nominal relationship. The dataset contains 97568 records in total of Dehumidifier Meter across 6 different stores. Data was cleaned by removing the values which are either 0 or lies above Mean + 2 * Standard Deviation. Data was shuffled to take the samples from complete dataset. Data was then split into Training Set 80%, Validation Set 10%, Testing Set 10%. Then a ANN model was trained using Power Consumption as the Output Variable. We trained a single hidden layer model with linear output layer. Adam optimizer was used with Mean Squared Error as the cost Function and a batch-size of 32. Earlystopping was used to avoid overfitting. The front-end was designed using Dash-plotly where the model results are displayed in form of Graphs. Actual Value lying outside 95% Confidence interval of predicted value was categorised as anomaly. 

Team

This project was designed during my Summer Internship at Walmart Labs in 2020 along with Ritish Menon and Mainak Mitra
 


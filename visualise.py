import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
from datetime import datetime
from tensorflow.keras import models
import os
from one_hot import one_hot


def read_data():
    df1=pd.read_csv('data_1.csv')
    df1=df1.rename(columns={'indoor_dewpoint_sensor':'ids','indoor_temperature_sensor':'its','outdoor_dewpoint_sensor':'ods',
    'outdoor_temperature_sensor':'ots','store_nbr':'s_n','meter_name':'m_n','time':'Timestamp'})
    df=pd.read_csv('data_2.csv')
    df=df.rename(columns={'indoor dewpoint sensor':'ids','indoor temperature sensor':'its','outdoor dewpoint sensor':'ods',
    'outdoor temperature sensor':'ots','store_nbr':'s_n','meter_name':'m_n','time':'Timestamp'})
    df=df.append(df1,ignore_index=True)

    return df
    
def update_data(df):
    timestamp=[]
    for d in range(len(df['Timestamp'])):
        date=df['Timestamp'][d].split('T')[0]+' '+df['Timestamp'][d].split('T')[1].split('.')[0]
        timestamp.append(datetime.strptime(date, "%Y-%m-%d %H:%M:%S"))
    df['Timestamp']=timestamp
    keys=[]
    df,key=one_hot.onehotencode(df,'dow','d')
    [keys.append(x) for x in key]
    df,key=one_hot.onehotencode(df,'hod','h')
    [keys.append(x) for x in key]
    df_original=df.copy()
    return df_original,keys

df=read_data()
df_original,keys=update_data(df)
outputs=['kwh']
inputs=['ids','its','ots','ods']
[inputs.append(x) for x in keys]

model_store={}
for m_n in os.listdir('models/'):
    model_store[m_n]={}
    for s_n in os.listdir('models/'+str(m_n)):
        model_store[m_n][s_n]=models.load_model('models/{}/{}'.format(m_n,s_n))

for m_n in model_store:
    for s_n in model_store[m_n])):
        df=df_original[df_original['m_n']==m_n].copy()
        df=df[df['s_n']==int(s_n)]
        df=df.sort_values('Timestamp')
        df.set_index(df['Timestamp'],inplace=True)
        actual_inputs=df[inputs]
        actual_outputs=pd.DataFrame(df['kwh'],df['Timestamp'])['kwh']
        predicted=model_store[m_n][s_n].predict(actual_inputs)
        predicted=pd.DataFrame(predicted,df['Timestamp'],columns=['kwh'])['kwh']
        predicted.index=df['Timestamp']
        anomalies=abs(actual_outputs-predicted)
        anomalies=actual_outputs[anomalies>2*(np.std(actual_outputs))]
        ids=pd.DataFrame(df['ids'],df['Timestamp'],columns=['ids'])['ids']
        its=pd.DataFrame(df['its'],df['Timestamp'],columns=['its'])['its']
        ods=pd.DataFrame(df['ods'],df['Timestamp'],columns=['ods'])['ods']
        ots=pd.DataFrame(df['ots'],df['Timestamp'],columns=['ots'])['ots']
        
        mp,(ax1,ax2,ax3,ax4,ax5)=plt.subplots(5)
        ax1.plot(ids,label='ids',color='r')
        ax2.plot(its,label='its',color='m')
        ax3.plot(ods,label='ods',color='c')
        ax4.plot(ots,label='ots',color='k')
        ax1.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()
        ax5.plot(actual_outputs,label='original',color='y')
        ax5.plot(predicted,label='predictions',color='b')
        ax5.scatter(anomalies.index,anomalies,color='g',label='anomaly')
        ax5.legend()
        plt.suptitle("Meter : {} Store : {}".format(m_n,s_n))
        plt.show()
        plt.close()

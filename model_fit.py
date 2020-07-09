import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.models import Sequential
import plotly.graph_objects as go
from tensorflow.keras.optimizers import Adam,SGD
import numpy as np
from matplotlib import pyplot as plt 
from datetime import datetime
import time
from one_hot import one_hot
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
import os

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

def data_clean(df):
    df=df.reset_index()
    kwh_new=df['kwh'].copy()
    kwh=df['kwh']
    last=0
    for y in range(1,len(kwh)-1):
        # remove if 0 or above mean + 2 * standard deviation
        if ((kwh[y]-np.mean(kwh))>(np.std(kwh)*2))or kwh[y]==0.0:
            kwh_new.pop(kwh.index[y])
        else:
            last=y

    df=df[df['kwh'].isin(kwh_new)]
    df=df.dropna()
    df.set_index(df['Timestamp'],inplace=True)
    df=df.sample(frac=1)
    return df

df=read_data()
df_original,keys=update_data(df)
outputs=['kwh']
inputs=['ids','its','ots','ods']
[inputs.append(x) for x in keys]

for m_n in np.unique(df_original['m_n']):
    for s_n in np.unique(df_original['s_n']):
        df=df_original[df_original['m_n']==m_n].copy()
        df=df[df['s_n']==s_n]
        df=data_clean(df)
        l=len(df)

        K.clear_session()
        x_train=df[inputs][:int(l*0.8)]
        x_validate=df[inputs][int(l*0.8):int(l*0.9)]
        x_test=df[inputs][int(l*0.9):]
        y_train=df[outputs][:int(l*0.8)]
        y_validate=df[outputs][int(l*0.8):int(l*0.9)]
        y_test=df[outputs][int(l*0.9):]

        x_train=np.array(x_train)
        x_validate=np.array(x_validate)
        x_test=np.array(x_test)
        y_train=np.array(y_train)
        y_validate=np.array(y_validate)
        y_test=np.array(y_test)
        callback=EarlyStopping(monitor='val_loss',patience=50,verbose=1,restore_best_weights=False)
        model=Sequential()
        model.add(Dense(20, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mean_squared_error',optimizer='Adam',metrics=['acc'])
        model.fit(x_train,y_train,validation_data=(x_validate,y_validate),epochs=1000,verbose=2,batch_size=32,callbacks=[callback])
        model.evaluate(x_test,y_test)
        plt.plot(y_test)
        plt.plot(model.predict(x_test))
        plt.show()
        if not os.path.exists('models/{}'.format(m_n)):
            os.mkdir('models/{}'.format(m_n))
        model.save('models/{}/{}'.format(m_n,s_n))

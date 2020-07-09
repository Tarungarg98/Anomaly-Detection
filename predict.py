### Importing the required Libraries

import plotly.graph_objs as go
import pandas as pd
from tensorflow.keras import models
import numpy as np
from datetime import datetime
import datetime as dt
import plotly.graph_objects as go
from one_hot import one_hot
import os

### Class object for reading and processing the data
class data_process:
    def __init__(self):
        ### Loading the models
        model_store={}
        for m_n in os.listdir('models/'):
            model_store[m_n]={}
            for s_n in os.listdir('models/'+str(m_n)):
                model_store[m_n][s_n]=models.load_model('models/{}/{}'.format(m_n,s_n))
        self.model=model_store
        
        ### Loading the data
    def load_data(self):
		df1=pd.read_csv('data_1.csv')
		df1=df1.rename(columns={'indoor_dewpoint_sensor':'ids','indoor_temperature_sensor':'its','outdoor_dewpoint_sensor':'ods', 'outdoor_temperature_sensor':'ots','store_nbr':'s_n','meter_name':'m_n','time':'Timestamp'})
		df=pd.read_csv('data_2.csv')
		df=df.rename(columns={'indoor dewpoint sensor':'ids','indoor temperature sensor':'its','outdoor dewpoint sensor':'ods',
		'outdoor temperature sensor':'ots','store_nbr':'s_n','meter_name':'m_n','time':'Timestamp'})
		df=df.append(df1,ignore_index=True)
        self.df=df
        
        ### Processing the data
    def pre_process(self):
        df=self.df
        timestamp=[]
        
        ### Converting Timestamp to DateTime
        for d in range(len(df['Timestamp'])):
            date=df['Timestamp'][d].split('T')[0]+' '+df['Timestamp'][d].split('T')[1].split('.')[0]
            timestamp.append(datetime.strptime(date,"%Y-%m-%d %H:%M:%S"))
        df['Timestamp']=timestamp
        keys=[]
        
        ### One-Hot encoding of Day of Week and Hour of Day 
        df,key=one_hot.onehotencode(df,'dow','d')
        [keys.append(x) for x in key]
        df,key=one_hot.onehotencode(df,'hod','h')
        [keys.append(x) for x in key]
        
        ### Dropping rows with null Values
        df=df.dropna()
        self.df=df
        self.keys=keys
        
    ### Defining input and output variables
    def parameters(self):
        df=self.df
        keys=self.keys
        inputs=['ids','its','ots','ods']
        [inputs.append(x) for x in keys]
        outputs='kwh'
        l=len(df)
        self.df_original=df.copy()
        self.inputs=inputs
        self.outputs=outputs

### Class Objects for each store
class prediction:
    def __init__(self,meter,store,base):
        self.meter=meter            # Meter Name
        self.store=store            # Store Number
        df_original=base.df_original            # DataSet
        self.model=base.model[meter][store]     # Model              
        self.inputs=base.inputs                 # Input Variable
        self.outputs=base.outputs               # Ouput Variable
        df=df_original[df_original['m_n']==meter].copy()       # Current Meter Data
        df=df[df['s_n']==int(store)]                        # Current Store Data
        df=df.sort_values('Timestamp')                      # Sort by TimeStamp
        df.set_index(df['Timestamp'],inplace=True)          # Setting Timestamp as index
        self.fault_key="Meter:"+str(meter)+", Store:"+str(store)
        self.df=df
        self.start_point=0                      # Setting starting row index set as 0
    
    def predict(self,all_faults,top_faults,current_faults,faulty_meters,faulty_stores,data_shift=10):
        start_point=self.start_point
        store=self.store
        inputs=self.inputs
        outputs=self.outputs
        model=self.model
        meter=self.meter
        df=self.df
        data_shift*=4  # 4 intervals in 1 hour
        fault_key=self.fault_key
        if data_shift!=0:    
            # Redefining data
            try:
                actual_inputs=df[inputs][start_point:start_point+data_shift]
                X_a=actual_inputs.index
                actual_outputs=pd.DataFrame(df[outputs][X_a[0]:X_a[-1]],df['Timestamp'][X_a[0]:X_a[-1]])['kwh']
            except Exception as e:
                start_point=0           #reset the graphs
                actual_inputs=df[inputs][start_point:start_point+data_shift]
                X_a=actual_inputs.index
                actual_outputs=pd.DataFrame(df[outputs][X_a[0]:X_a[-1]],df['Timestamp'][X_a[0]:X_a[-1]])['kwh']
            start_point+=data_shift
            self.start_point=start_point    
            actual_outputs_anomaly=pd.DataFrame(df[outputs],df['Timestamp'])['kwh']
            predicted=model.predict(actual_inputs)   #model predictions
            predicted=pd.DataFrame(predicted,df['Timestamp'][X_a[0]:X_a[-1]],columns=[outputs])[outputs]
            predicted.index=df['Timestamp'][X_a[0]:X_a[-1]]
            anomalies=abs(actual_outputs-predicted)
            anomalies_value=anomalies[anomalies>2*np.std(actual_outputs_anomaly)]
            anomalies=actual_outputs[anomalies>2*np.std(actual_outputs_anomaly)]
            if len(anomalies_value)>0:
                fault_df=pd.DataFrame()
                fault_df['Timestamp']=anomalies_value.index
                fault_df['Meter']=fault_key
                fault_df['Value']=[round(v,2) for v in anomalies_value]
                all_faults=all_faults.append(fault_df,ignore_index=True)
                current_faults=current_faults.append({'Meter':fault_key},ignore_index=True)
                faulty_meters.append({'label':str(meter),'value':str(meter)})
                faulty_stores.append({'label':str(store),'value':str(store)})
            del model
            try:
                self.X_inputs=self.X_inputs.append(actual_inputs)
                self.X=self.X.append(X_a)
                self.Y_a=self.Y_a.append(actual_outputs)
                self.Y_p=self.Y_p.append(predicted)
                self.Y_anomaly=self.Y_anomaly.append(anomalies)
                self.X_anomaly=self.X_anomaly.append(anomalies.index)
            except:
                self.X_inputs=actual_inputs
                self.X=X_a
                self.Y_a=actual_outputs
                self.Y_p=predicted
                self.Y_anomaly=anomalies
                self.X_anomaly=anomalies.index
            if start_point==0:
                self.X_inputs=actual_inputs
                self.X=X_a
                self.Y_a=actual_outputs
                self.Y_p=predicted
                self.Y_anomaly=anomalies
                self.X_anomaly=anomalies.index
        self.start_point=start_point    
        #for top faults in last 7 days
        old_date=max(self.X)-dt.timedelta(days=7)
        tmp=all_faults.loc[all_faults['Meter']==fault_key].copy()
        top_faults=top_faults.append(tmp.loc[all_faults['Timestamp']>=old_date].sort_values(['Value'],
        ascending=False).reset_index(drop=True),ignore_index=True)
        return all_faults,top_faults,current_faults,faulty_meters,faulty_stores
    
    def graph_all(self,no_of_data_points=100):
        no_of_data_points*=4
        store=self.store
        meter=self.meter
        X_a=self.X[-no_of_data_points:]
        Y_a=np.array(self.Y_a[-no_of_data_points:])
        Y_p=np.array(self.Y_p[-no_of_data_points:])
        Y_anomaly=np.array(self.Y_anomaly[-no_of_data_points:])
        X_anomaly=self.X_anomaly[-no_of_data_points:]
        tmp=[]
        tmp2=[]
        # Choosing anomalies in current timeframe
        for v in range(len(X_anomaly)):
            value=X_anomaly[v]
            if value in X_a:
                tmp.append(value)
                tmp2.append(Y_anomaly[v])
        X_anomaly=tmp
        Y_anomaly=tmp2
        data=[
        go.Scatter(x=X_a,y=Y_a,name='Original',mode='lines'),
        go.Scatter(x=X_a,y=Y_p,name='Predicted',mode='lines')
        ]
        if len(Y_anomaly)>0:
            data.append(go.Scatter(x=X_anomaly,y=Y_anomaly,name='Anomaly',mode='markers',marker=dict(color='Red',size=8)))
        xaxis=dict(range=[X_a[0],X_a[-1]])
        yaxis=dict(range=[0,max(np.max(Y_a),np.max(Y_p),30)+1])
        figure={'data':data,'layout':go.Layout(xaxis=xaxis,
                yaxis=yaxis,title="Store :{}".format(store))}
        return figure
    
    def graph_analyse(self,start_date,end_date):
        store=self.store
        meter=self.meter
        start_date=datetime.strptime(start_date,"%Y-%m-%d")    
        end_date=datetime.strptime(end_date,"%Y-%m-%d")    
        X_a=[]
        Y_a=[]
        Y_p=[]
        ids=[]
        ods=[]
        its=[]
        ots=[]
        X_anomaly=[]
        Y_anomaly=[]
        for x in range(len(self.X)):
            a,b,c,d,e,f,g = self.X[x],np.array(self.Y_a)[x],np.array(self.Y_p)[x],self.X_inputs['ids'][x],self.X_inputs['its'][x],self.X_inputs['ods'][x],self.X_inputs['ots'][x]
            if a>=start_date and a<=end_date:
                    X_a.append(a)
                    Y_a.append(b)
                    Y_p.append(c)
                    ids.append(d)
                    its.append(e)
                    ods.append(f)
                    ots.append(g)
        for x in range(len(self.X_anomaly)):
            a,b = self.X_anomaly[x],np.array(self.Y_anomaly)[x]
            if a>=start_date and a<=end_date:
                    X_anomaly.append(a)
                    Y_anomaly.append(b)
                   
                

        data=[
        go.Scatter(x=X_a,y=ids,name='ids',mode='lines'),
        go.Scatter(x=X_a,y=its,name='its',mode='lines'),
        go.Scatter(x=X_a,y=ods,name='ods',mode='lines'),
        go.Scatter(x=X_a,y=ots,name='ots',mode='lines'),
        go.Scatter(x=X_a,y=Y_a,name='Original',mode='lines'),
        go.Scatter(x=X_a,y=Y_p,name='Predicted',mode='lines')
        ]
        xaxis=dict(range=[X_a[0],X_a[-1]])        
        yaxis=dict(range=[0,max(np.max(Y_a),np.max(Y_p),np.max(ids),np.max(its),
        np.max(ods),np.max(ots),30)+1])
        if len(Y_anomaly)>0:
            data.append(go.Scatter(x=X_anomaly,y=Y_anomaly,name='Anomaly',mode='markers',marker=dict(color='Red',size=8)))
        figure={'data':data,'layout':go.Layout(xaxis=xaxis,
                yaxis=yaxis,title="Store :{}".format(store))}
        return figure
    

import dash
from dash.dependencies import Output , Input
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import math
from datetime import datetime
import dash_table

import pandas as pd
import plotly.graph_objects as go

import numpy as np
from one_hot import one_hot
from predict import prediction,data_process

def create_base():
    ob=data_process()
    ob.load_data()
    ob.pre_process()
    ob.parameters()
    return ob
    
base=create_base()    
    
ob110=prediction("DEHUMID 2","110",base)
ob130=prediction("DEHUMID 2","130",base)
ob2464=prediction("DEHUMID 2","2464",base)
ob3570=prediction("DEHUMID 2","3570",base)
ob5964=prediction("DEHUMID 2","5964",base)
start_point=0

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
stores=[110,130,2464,3570,5964]
meters=["DEHUMID 2"]

app.layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                
                    html.Div([
                    html.I("Power Consumption\t",style={'font-size':60}),
                    html.Br(),
                    html.I("Refresh-Rate (in seconds)\t"),
                    dcc.Input(value=5, id='refresh',type='number',placeholder="Refresh-rate"),
                    html.I("\t"),
                    html.Button('Pause', id='pause', n_clicks=0,style={"align":"right"}),
    dcc.Tabs([
        dcc.Tab(label='All Meters', value='tab-1',children=[
                    html.I("Meter"),
                    dcc.Dropdown(id="Meter Name",options=[{'label':str(m),'value':str(m)} for m in meters]
                    ,value=[],multi=True),
                    html.I("Meter"),
                    dcc.Dropdown(id="Store Number",options=[{'label':str(s),'value':str(s)} for s in stores]
                    ,value=[],multi=True),
                    html.Div(children=[
                    # html.H2(id='errors'),
                    html.I("Enter no of hours to be displayed\t"),
                    dcc.Input(id='no_of_data_points',value=168,type='number',placeholder="No of records"),
                    html.Br(),
                    html.I("Enter no of hours to shift the data\t"),
                    dcc.Input(id='data_shift',value=24,type='number',placeholder="Shift by"),
                    html.Div(id='all-graphs')]),
        ]),
        dcc.Tab(label='Faulty Meters', value='tab-2',children=[
                    dcc.Dropdown(id="Faulty Meter Name",value=[],multi=True),
                    dcc.Dropdown(id="Faulty Store Number",value=[],multi=True),
                    html.Div(children=[
                    html.Div(id='faulty-graphs')]),

    ]),                    
        dcc.Tab(label='Analyze Meters', value='tab-6',children=[
            dcc.Dropdown(id="Analyse Meter Name",options=[{'label':str(m),'value':str(m)} for m in meters],value=[],multi=True),
            dcc.Dropdown(id="Analyse Store Number",options=[{'label':str(s),'value':str(s)} for s in stores],value=[],multi=True),
            html.I("Date Range\t"),
            dcc.DatePickerRange(
                    id='my-date-picker-range',
                    start_date=datetime(2019, 6, 1).date(),
                    end_date=datetime(2020, 4, 25).date()
                ),
            html.I("\t"),
            html.Button('Update', id='update', n_clicks=0,style={"align":"right"}),
            html.Div(children=[html.Div(id='analyse-graphs')]),

    ])                    

    ]),                    
                    dcc.Interval(id='graph-update',interval=5000)]),

                    style={"height": "100%"},
                    width=8,
                ),
                dbc.Col(
    dcc.Tabs([
        dcc.Tab(label="Alerts", value='tab-3',children=[
                    html.Div(id="current_faults")
                    ]),
        dcc.Tab(label="Faults in last 1 week", value='tab-4',children=[
                    html.Div(id="top_faults")
                    ]),
                    dcc.Tab(label='All Faults', value='tab-5',children=[
                    html.Div(id="all_faults")
                    ])]),
                    width=4,
                    style={"height": "100%", "background-color": "#D9D9D9"},
                ),
            ],
            className="h-100 w-100",
        )
    ],
    style={"height": "100vh","width": "100vw","margin-left":"10px"},
)
all_faults=pd.DataFrame(columns=["Timestamp",'Meter','Value'])
objects={'DEHUMID 2':[ob110,ob130,ob2464,ob3570,ob5964]}
# objects={'DEHUMID 2':[ob110]}

@app.callback([
Output('graph-update','interval')],
[Input('refresh','value'),])
def refresh_rate(value):
    value=int(value)
    if value<1:
        value=1
    
    return [value*1000]

@app.callback([
Output('graph-update','disabled'),
Output('pause','children')],
[Input('pause','n_clicks'),])
def update_interval(n_clicks):
    update_state=False
    button_name="Pause"
    if n_clicks%2!=0:
        button_name="Resume"
        update_state=True
    return update_state,button_name




@app.callback([
Output("current_faults",'children'),
Output('top_faults','children'),
Output('all_faults','children'),
Output('Faulty Meter Name', 'options'),
Output('Faulty Store Number', 'options')
],
[
Input('Store Number', 'value'),
Input('Meter Name', 'value'),
Input('no_of_data_points','value'),
Input('data_shift','value'),
Input('graph-update', 'n_intervals')])
def update_data(store_numbers,meter_names,no_of_data_points,data_shift,interval):

    top_faults=pd.DataFrame(columns=["Timestamp",'Meter','Value'])

    # current_faults=pd.DataFrame(columns=["Timestamp",'Meter'])
    current_faults=pd.DataFrame(columns=['Meter'])
    graphs=[]
    global stores
    global start_point
    faulty_meters=[]
    faulty_stores=[]
    global all_faults
    try:
        data_shift+10
    except:
        data_shift=0
    # All Meters   
    for meter_name in meters:
        for store_id in range(len(objects[meter_name])):
            store=stores[store_id]
            all_faults,top_faults,current_faults,faulty_meters,faulty_stores=objects[meter_name][store_id].predict(
            all_faults,top_faults,current_faults,faulty_meters,faulty_stores,data_shift)
    tmp=faulty_meters.copy()
    faulty_meters=[]
    for x in tmp:
        if x not in faulty_meters:
            faulty_meters.append(x)
    tmp=faulty_stores.copy()
    faulty_stores=[]
    # Storing faulty meters deatils for faulty graphs
    for x in tmp:
        if x not in faulty_stores:
            faulty_stores.append(x)
    top_faults_table=dash_table.DataTable(columns=[{"name": i, "id": i,} for i in (top_faults.columns)],
    style_cell={'textAlign': 'left','padding':'1px'},style_data_conditional=[
        {'if': {'row_index': 'odd'},'backgroundColor': 'rgb(248, 248, 248)'}],
    data=top_faults.to_dict('records'))
    all_faults_table=dash_table.DataTable(columns=[{"name": i, "id": i,} for i in (all_faults.columns)],
    style_cell={'textAlign': 'left','padding':'1px'},style_data_conditional=[
        {'if': {'row_index': 'odd'},'backgroundColor': 'rgb(248, 248, 248)'}],
    data=all_faults.to_dict('records'))
    current_faults_table=dash_table.DataTable(columns=[{"name": i, "id": i,} for i in (current_faults.columns)],
    style_cell={'textAlign': 'left','padding':'1px'},style_data_conditional=[
        {'if': {'row_index': 'odd'},'backgroundColor': 'rgb(248, 248, 248)'}],
    data=current_faults.to_dict('records'))
    return current_faults_table,top_faults_table,all_faults_table,faulty_meters,faulty_stores

@app.callback([
Output('all-graphs','children'),
Output('faulty-graphs','children'),
],
[
Input('Store Number', 'value'),
Input('Meter Name', 'value'),
Input('Faulty Store Number', 'value'),
Input('Faulty Meter Name', 'value'),
Input('no_of_data_points','value'),
Input('graph-update', 'n_intervals')])
def display_graph(store_numbers,meter_names,faulty_store_numbers,faulty_meter_names,no_of_data_points,interval):
    graphs_all=[]
    graphs_faulty=[]
    global stores

    for meter_name in meters:
        for store_id in range(len(objects[meter_name])):
            store=stores[store_id]
            if str(store) in store_numbers and meter_name in meter_names:
                figure=objects[meter_name][store_id].graph_all(no_of_data_points=no_of_data_points)
                graphs_all.append(html.Div(dcc.Graph(animate=True,figure=figure)))
            if str(store) in faulty_store_numbers and meter_name in faulty_meter_names:
                figure=objects[meter_name][store_id].graph_all(no_of_data_points=no_of_data_points)
                graphs_faulty.append(html.Div(dcc.Graph(animate=True,figure=figure)))
    return graphs_all,graphs_faulty




 
@app.callback([
Output('analyse-graphs','children'),
],
[
Input('Analyse Store Number', 'value'),
Input('Analyse Meter Name', 'value'),
Input('my-date-picker-range', 'start_date'),
Input('my-date-picker-range', 'end_date'),
Input('update','n_clicks')])
def analyse_graph(store_numbers,meter_names,start_date,end_date,n_clicks):
    graphs_analyse=[]
    global stores

    for meter_name in meters:
        for store_id in range(len(objects[meter_name])):
            store=stores[store_id]
            if str(store) in store_numbers and meter_name in meter_names:
                try:
                    fig=objects[meter_name][store_id].graph_analyse(start_date,end_date)
                    graphs_analyse.append(html.Div(dcc.Graph(animate=True,figure=fig)))
                except:
                    pass
    return [graphs_analyse]

 
if __name__=="__main__":
    app.run_server(debug=True,port=8050)
    
    
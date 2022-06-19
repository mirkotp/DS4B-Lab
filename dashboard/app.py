from audioop import mul
from itertools import permutations
from math import perm
from pydoc import classname
from re import T
import re
import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html
import plotly.graph_objects as go
import dash_daq as daq
from random import random
import numpy as np
import flask
import pandas as pd
import os

server = flask.Flask('app')
server.secret_key = os.environ.get('secret_key', 'secret')
app = dash.Dash('app', server=server)


# LOAD DATA
permanent_jobs = pd.read_csv("~/data/permanent_jobs.csv")

# LAYOUT
app.layout = html.Div(
    children=[ 
        html.Div(
            className="header",
            children=[
                html.H1(
                    children="💉 Covid-19 Impact in Lombardy's Job Market 🦠",
                    className="header-title"
                ),
                html.P(
                    children="Let's analyze the impact of Covid-19 pandemic on the drafting of new \"Tempo Indeterminato\" contracts.",
                    className="header-description"
                ),
            ]
        ),

        dcc.Tabs(
            id="tabs",
            value='tab-1', 
            children=[
                dcc.Tab(
                    label='Predict', 
                    value='tab-1',
                    children=[
                        dcc.Loading(
                            id="pred-loading",
                            type="default",
                            children=[
                                html.Div(
                                    className="chart-inputs",
                                    children=[
                                        html.Div(
                                            className="cats-checklist",
                                            children=[
                                                "Età:",
                                                dcc.Dropdown(
                                                    ['-'] + list(sorted(permanent_jobs["ETA"].unique())),
                                                    value='-',
                                                    id="pred-eta"
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            className="cats-checklist",
                                            children=[
                                                "Genere:",
                                                dcc.Dropdown(
                                                    ['-'] + list(sorted(permanent_jobs["GENERE"].unique())),
                                                    value='-',
                                                    id="pred-genere"
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            className="cats-checklist",
                                            children=[
                                                "Provincia:",
                                                dcc.Dropdown(
                                                    ['-'] + list(sorted(permanent_jobs["PROVINCIAIMPRESA"].unique())),
                                                    value='-',
                                                    id="pred-provinciaimpresa"
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            className="cats-checklist",
                                            children=[
                                                "Istruzione:",
                                                dcc.Dropdown(
                                                    ['-'] + list(sorted(permanent_jobs["TITOLOSTUDIO"].unique())),
                                                    value='-',
                                                    id="pred-titolostudio"
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            className="cats-checklist",
                                            children=[
                                                "Italiano:",
                                                dcc.Dropdown(
                                                    ['-'] + list(sorted(permanent_jobs["ITALIANO"].unique())),
                                                    value='-',
                                                    id="pred-italiano"
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            className="cats-checklist",
                                            children=[
                                                "Macrosettore:",
                                                dcc.Dropdown(
                                                    [{'label':'-', 'value': '-'}] +  [
                                                            {
                                                                'label': (str(x)[:40] + '...') if len(str(x)) > 40 else str(x),
                                                                'value': str(x)
                                                            }  for x in sorted(permanent_jobs["GRUPPOSETTORE"].unique())
                                                        ],
                                                    value='-',
                                                    id="pred-grupposettore"
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            className="cats-checklist",
                                            children=[
                                                "Model:",
                                                dcc.Dropdown(
                                                    ['SARIMA', 'LSTM'],
                                                    value='SARIMA',
                                                    id="pred-model"
                                                ),
                                            ]
                                        ),
                                        html.Button(
                                            'RUN', 
                                            id='pred-submit', 
                                            n_clicks=0)
                                    ],
                                ),
                                html.Div(
                                    children=[
                                        dcc.Graph(
                                            id="pred-graph",
                                        ),
                                        html.Div(
                                            id="pred-info"
                                        )
                                    ]
                                ),
                                
                            ]
                        ),
                    ]
                ),
                dcc.Tab(
                    label='Compare by attribute(s)',
                    value='tab-2',
                    children=[
                        dcc.Loading(
                            id="comp-loading",
                            type="default",
                            children=[
                                html.Div(
                                    className="chart-inputs",
                                    children=[
                                        html.Div(
                                            className="cats-checklist",
                                            children=[
                                                dcc.Checklist(
                                                    [
                                                        {'label': 'Età', 'value': 'ETA'},
                                                        {'label': 'Genere', 'value': 'GENERE'},
                                                        {'label': 'Provincia', 'value': 'PROVINCIAIMPRESA'},
                                                        {'label': 'Istruzione', 'value': 'TITOLOSTUDIO'},
                                                        {'label': 'Italiano', 'value': 'ITALIANO'},
                                                        {'label': 'Macrosettore', 'value': 'GRUPPOSETTORE'}
                                                    ],
                                                    [],
                                                    id="comp-cats",
                                                    inline=True
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            className="cats-checklist",
                                            children=[
                                                "Model:",
                                                dcc.Dropdown(
                                                    ['SARIMA', 'LSTM'],
                                                    value='SARIMA',
                                                    id="comp-model"
                                                ),
                                            ]
                                        ),
                                        daq.BooleanSwitch(
                                            id="comp-sort",
                                            label="Sort by Y",
                                            labelPosition="bottom",
                                            on=True,
                                            color="#9B51E0",
                                        ),
                                        daq.BooleanSwitch(
                                            id="comp-standardize",
                                            label="Standardize",
                                            labelPosition="bottom",
                                            on=True,
                                            color="#9B51E0",
                                        ),
                                        html.Button(
                                            'RUN', 
                                            id='comp-submit', 
                                            n_clicks=0
                                        )
                                    ],
                                ),
                                html.Div(
                                    children=dcc.Graph(
                                        id="comp-graph",
                                    ),
                                ),
                                
                            ]
                        )
                    ]
                ),
            ]
        ),
    ]
)

# APP LOGICS

#SARIMA
from pmdarima.arima import auto_arima, ndiffs, nsdiffs
from statsmodels.tsa.api import SARIMAX
def SARIMA_predict(ts, column=0, fast=False, standardize=False):
    # Reindex     
    ts = ts.reindex([(x, y) for x in range(2010, 2022) for y in range(1, 13)], fill_value=0)
    
    # Time series standardization
    if(standardize):
        ts = (ts - ts.min()) / ts.std()
    
    # Adjust dataset format
    ts = ts.reset_index().reset_index()
    ts["DATA"] = ts["ANNO"].astype(str) + "/" + ts["MESE"].astype(str) + "/01"
    ts["DATA"] = pd.to_datetime(ts["DATA"])
    ts = ts.set_index("DATA")[column]
    ts = ts.fillna(0)
    ts = ts.rename('y')

    # Remove covid period
    ts_no_covid = ts[:'2020-01-01']
    model = auto_arima(   ts_no_covid,
                          information_criterion='aic',
                          test='adf',
                          max_p=3,
                          max_q=3,
                          d=ndiffs(ts_no_covid),
                          D=nsdiffs(ts_no_covid, 12),
                          method='nm' if fast else 'lbfgs',
                          seasonal=True,
                          m=12,
                          trace=True,
                          error_action="ignore",
                          suppress_warnings=True,
                          stepwise=True,
                          with_intercept=True)
    results = model.fit_predict(ts_no_covid, n_periods=24, return_conf_int=True)
    orderStr = f"{model.order}{model.seasonal_order}{' with intercept' if model.with_intercept else '' }"
        
    return ts['2020-01-01':].sum() - results[0].sum(), results, orderStr

# LSTM

import multiprocessing
from multiprocessing import Process

def LSTM_predict(ts, column=0, standardize=False):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    p = Process(target=LSTM_predict_p, args=(return_dict, ts, column, standardize))
    p.start()
    p.join() 

    return return_dict['return']

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
def LSTM_predict_p(return_dict, ts, column=0, standardize=False, ):
# Time series standardization
    if(standardize):
        ts = (ts - ts.min()) / ts.std() 

    np.random.seed(42)
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Adjust dataset format
    months = [(x, y) for x in range(2010, 2022) for y in range(1, 13)]
    ts = ts.reindex(months, fill_value=0)

    for _ in range(ts.index.nlevels - 2):
            ts = ts.unstack()

    ts = ts.reset_index().reset_index()
    ts["DATA"] = ts["ANNO"].astype(str) + "/" + ts["MESE"].astype(str) + "/01"
    ts["DATA"] = pd.to_datetime(ts["DATA"])
    ts = ts.set_index("DATA")[column]
    ts = ts.fillna(0)

    ts_no_covid = ts[:'2020-01-01']
    ts_no_covid = scaler.fit_transform(ts_no_covid.values.astype('float32').reshape(-1, 1))

    train_size = int(len(ts_no_covid) * 0.75)
    train, test = ts_no_covid[0:train_size,:], ts_no_covid[train_size:len(ts_no_covid),:]


    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    # Create and reshape datasets
    look_back = 24
    trainX, trainY = create_dataset(train, look_back)
    testX, _ = create_dataset(test,look_back)
    tsX, _ = create_dataset(ts_no_covid, look_back)
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    tsX = np.reshape(tsX, (tsX.shape[0], 1, tsX.shape[1]))

    # LSTM model
    model = Sequential()
    model.add(LSTM(16, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=30, batch_size=1, verbose=0)

    # Prediction
    try:
        prediction = model.predict(tsX)
        prediction = scaler.inverse_transform(prediction)[:,0]
        prediction = pd.Series(prediction[-24:], ts['2020-01-01':].index)
    except Exception as e: 
            print(e)
            return [], 0
        
    pred = prediction
    diff = ts['2020-01-01':].sum() - prediction.sum()

    return_dict['return'] = (pred, diff)

# PREDICT CALLBACK
def predict_ts_sarima(eta, genere, provinciaimpresa, titolostudio, italiano, grupposettore):
    allTrue = permanent_jobs["ETA"] == permanent_jobs["ETA"] 

    ds = permanent_jobs[
        (permanent_jobs["ETA"] == eta if eta != '-' else allTrue) & 
        (permanent_jobs["GENERE"] == genere if genere != '-' else allTrue) &
        (permanent_jobs["PROVINCIAIMPRESA"] == provinciaimpresa if provinciaimpresa != '-' else allTrue) &
        (permanent_jobs["TITOLOSTUDIO"] == titolostudio if titolostudio != '-' else allTrue) &
        (permanent_jobs["ITALIANO"] == italiano if italiano != '-' else allTrue) &
        (permanent_jobs["GRUPPOSETTORE"] == grupposettore if grupposettore != '-' else allTrue)
    ]

    ds = ds.groupby(["ANNO", "MESE"]).size()
    ds = ds.reindex([(x, y) for x in range(2010, 2022) for y in range(1, 13)], fill_value=0)
    diff, pred, order = SARIMA_predict(ds)

    x = [str(x) for x in ds.index.to_list()]
    y = ds.to_list()

    pred_x = [str(x) for x in ds[(2020,1):].index.to_list()]
    pred_y = pred[0]

    upper_y = [x[1] for x in pred[1]]
    lower_y = [x[0] for x in pred[1]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, 
        y=y,
        name='Series'
    ))

    fig.add_trace(go.Scatter(
        x=[str(x) for x in ds[(2019,12):(2020,2)].index.to_list()], 
        y=[ds[(2019,12):].to_list()[0], pred_y[0]],
        fill=None,
        line={'dash':'dot', 'color': 'red'},
        mode='lines',
        name='SARIMA',
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=[str((2019,12))] + pred_x, 
        y=[ds[(2019,12):].to_list()[0]] + upper_y,
        fill=None,
        mode='lines',
        line_color='orange',
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=[str((2019,12))] + pred_x, 
        y=[ds[(2019,12):].to_list()[0]] + lower_y,
        fill='tonexty',
        mode='lines',
        line_color='orange',
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=pred_x, 
        y=pred_y,
        fill=None,
        mode='lines',
        line_color='red',
        name='SARIMA'
    ))

    return [fig, [
        html.Div(children=f"Difference: {round(diff, 2)}"), 
        html.Div(children=f"SARIMA order: {order}")
    ]]


def predict_ts_lstm(eta, genere, provinciaimpresa, titolostudio, italiano, grupposettore):
    allTrue = permanent_jobs["ETA"] == permanent_jobs["ETA"] 

    ds = permanent_jobs[
        (permanent_jobs["ETA"] == eta if eta != '-' else allTrue) & 
        (permanent_jobs["GENERE"] == genere if genere != '-' else allTrue) &
        (permanent_jobs["PROVINCIAIMPRESA"] == provinciaimpresa if provinciaimpresa != '-' else allTrue) &
        (permanent_jobs["TITOLOSTUDIO"] == titolostudio if titolostudio != '-' else allTrue) &
        (permanent_jobs["ITALIANO"] == italiano if italiano != '-' else allTrue) &
        (permanent_jobs["GRUPPOSETTORE"] == grupposettore if grupposettore != '-' else allTrue)
    ]


    ds = ds.groupby(["ANNO", "MESE"]).size()
    ds = ds.reindex([(x, y) for x in range(2010, 2022) for y in range(1, 13)], fill_value=0)
    pred, diff = LSTM_predict(ds)

    x = [str(x) for x in ds.index.to_list()]
    y = ds.to_list()

    pred_x = [str(x) for x in ds[(2020,1):].index.to_list()]
    pred_y = pred

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, 
        y=y,
        name='Series'
    ))

    fig.add_trace(go.Scatter(
        x=[str(x) for x in ds[(2019,12):(2020,2)].index.to_list()], 
        y=[ds[(2019,12):].to_list()[0], pred_y[0]],
        fill=None,
        line={'dash':'dot', 'color': 'red'},
        mode='lines',
        showlegend=False
    ))


    fig.add_trace(go.Scatter(
        x=pred_x, 
        y=pred_y,
        fill=None,
        mode='lines',
        line_color='red',
        name='LSTM' 
    ))

    return [fig, [f"Difference: {diff}"]]

@app.callback([Output('pred-graph', 'figure'), Output('pred-info', 'children')],
              [Input('pred-submit', 'n_clicks')],
              [State('pred-model', 'value'), State('pred-eta', 'value'), State('pred-genere', 'value'), State('pred-provinciaimpresa', 'value'), 
              State('pred-titolostudio', 'value'), State('pred-italiano', 'value'), State('pred-grupposettore', 'value'),])
def predict_callback(n_clicks, model, eta, genere, provinciaimpresa, titolostudio, italiano, grupposettore):
    if(n_clicks > 0):
        if(model == 'SARIMA'):
            return predict_ts_sarima(eta, genere, provinciaimpresa, titolostudio, italiano, grupposettore)
        elif(model == 'LSTM'):
            return predict_ts_lstm(eta, genere, provinciaimpresa, titolostudio, italiano, grupposettore)

    return [go.Figure(), ""]


# COMPARE CALLBACK
def compare_ts_sarima(sort=True, attrs=[], standardize=True):
    ds = permanent_jobs.groupby(["ANNO", "MESE"] + attrs).size()
    for _ in range(ds.index.nlevels - 2):
        ds = ds.unstack()

    results = []
    for col in pd.DataFrame(ds).columns:
        try:
            print(col)
            diff, _, _ = SARIMA_predict(ds, fast=True, standardize=standardize, column=col)
            results.append((col, diff))
        except Exception as e: 
            print(e)
            results.append((None, None))

    if(sort):
        results = sorted(results,  key=lambda x: x[1])

    results = list(zip(*results))
    x = [(str(x)[:50] + '...') if len(str(x)) > 50 else str(x) for x in results[0]]
    y = results[1]

    return [{
        "data": [
            {
                "x": x,
                "y": y,
                "type": "scatter",
            },
        ],
        "layout": {
            "title": {
                "x": 0.05,
                "xanchor": "left"
            },
            "xaxis": {"fixedrange": True},
            "yaxis": {"fixedrange": True},
            "colorway": ["#E12D39"],
        },
    }]

def compare_ts_lstm(sort=True, attrs=[], standardize=True):
    ds = permanent_jobs.groupby(["ANNO", "MESE"] + attrs).size()
    for _ in range(ds.index.nlevels - 2):
        ds = ds.unstack()

    import time
    results = []
    for col in pd.DataFrame(ds).columns:
        try:
            print(col)
            _, diff = LSTM_predict(ds, standardize=standardize, column=col)
            results.append((col, diff))
        except Exception as e: 
            print(e)
            results.append((None, None))

    if(sort):
        results = sorted(results,  key=lambda x: x[1])

    results = list(zip(*results))
    x = [(str(x)[:50] + '...') if len(str(x)) > 50 else str(x) for x in results[0]]
    y = results[1]

    return [{
        "data": [
            {
                "x": x,
                "y": y,
                "type": "scatter",
            },
        ],
        "layout": {
            "title": {
                "x": 0.05,
                "xanchor": "left"
            },
            "xaxis": {"fixedrange": True},
            "yaxis": {"fixedrange": True},
            "colorway": ["#E12D39"],
        },
    }]



@app.callback([Output('comp-graph', 'figure')],
            [Input('comp-submit', 'n_clicks')],
            [State('comp-model', 'value'), State('comp-sort', 'on'), State('comp-cats', 'value'), State('comp-standardize', 'on')])
def compare_callback(n_clicks, model, sort, attrs, standardize):
    if(n_clicks > 0):
        if(model == 'SARIMA'):
            return compare_ts_sarima(sort=sort, attrs=attrs, standardize=standardize)
        elif(model == 'LSTM'):
            return compare_ts_lstm(sort=sort, attrs=attrs, standardize=standardize)

    return [go.Figure()]

if __name__ == '__main__':
    app.run_server(debug=False)



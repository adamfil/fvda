# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from faulthandler import disable
from statistics import mean
from dash import Dash, html, dcc, callback_context, dash_table
from dash.dependencies import Input, Output, State
import dash_leaflet as dl
import plotly.express as px
import pandas as pd
import requests 
import numpy as np
from statsmodels.tools.eval_measures import rmse
from celery import Celery 
from dash.long_callback import CeleryLongCallbackManager
from datetime import date
import datetime
import plotly.io as pio
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import algo
from algo import MAP_METRICS, ValidationQuery
import dash_helper
from climatological_averager import climatological_averager

## new import 
import json
import math 

theme = [dbc.themes.SIMPLEX]

celery_app = Celery(
    __name__,
    broker = "redis://:pfd1c23f2b9201f4a0869041e17955284914f0135bbc7eaeb86b2aceb57556c3c@ec2-44-208-193-34.compute-1.amazonaws.com:19130/0",
    backend = "redis://:pfd1c23f2b9201f4a0869041e17955284914f0135bbc7eaeb86b2aceb57556c3c@ec2-44-208-193-34.compute-1.amazonaws.com:19130/1"
    #broker="redis://localhost:6379/0", backend="redis://localhost:6379/1"
)
long_callback_manager = CeleryLongCallbackManager(celery_app)

app = Dash(__name__, long_callback_manager=long_callback_manager, prevent_initial_callbacks=True, external_stylesheets=theme)
server = app.server

pio.templates.default = 'simple_white'

def round_decimals_up(number:float, decimals:int=2):
    """
    Returns a value rounded up to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.ceil(number)

    factor = 10 ** decimals
    return math.ceil(number * factor) / factor

def round_decimals_down(number:float, decimals:int=2):
    """
    Returns a value rounded down to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.floor(number)

    factor = 10 ** decimals
    return math.floor(number * factor) / factor

def slice_per(source, step):
    return [source[i::step] for i in range(step)]

polylines = []

'''
#polylines for lat/long grid (need to go in and fix 181 lats + 360 longs)
lats = [] 
longs = []

for i in range(1441):
    lats.append(i/4)
    longs.append(i/4 - 180)
for i in range(len(lats)):
    polylines.append(dl.Polyline(positions=[[0, longs[i]], [360, longs[i]]], color="#FF7F7F", opacity=0.15 ))
    polylines.append(dl.Polyline(positions=[[lats[i], 180], [lats[i], -180]], color="#FF7F7F", opacity=0.15))
'''
TOKEN = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjI1MzQwMjMwMDc5OSwiaWF0IjoxNjMwNTMwMTgwLCJzdWIiOiJhMDIzYjUwYi0wOGQ2LTQwY2QtODNiMS1iMTExZDA2Mzk1MmEifQ.qHy4B0GK22CkYOTO8gsxh0YzE8oLMMa6My8TvhwhxMk'

def get_date_range(dataset: str):
    my_url = 'https://api.dclimate.net/apiv3/metadata/' + dataset + '?full_metadata=true'
    head = {"Authorization": TOKEN}
    r = requests.get(my_url, headers=head)
    if 'api documentation' in r.json().keys():
        data = r.json()['api documentation']
        if 'full date range' in data.keys():
            print(f"Getting date range of {dataset}")
            start_date = data['full date range'][0]
            start_date = start_date.split('-')
            year_start = int(start_date[0])
            month_start = int(start_date[1])
            day_start = start_date[2].split(' ')
            day_start = int(day_start[0])
            start_date = datetime.datetime(year_start, month_start, day_start)

            end_date = data['full date range'][1]
            end_date = end_date.split('-')
            year_end = int(end_date[0])
            month_end = int(end_date[1])
            day_end = end_date[2].split(' ')
            day_end = int(day_end[0])
            end_date = datetime.datetime(year_end, month_end, day_end)

            return [start_date, end_date]

        else:
            return [datetime.datetime(2000, 1, 1), datetime.datetime.today()]
        #deal with gridhistory datasets that dont have API documentation for daterange in metadata
    else:
        return [datetime.datetime(2000, 1, 1), datetime.datetime.today()]


def date_range_extreme(lst: list):
    #input is list of forecasts/datasets 
    min_dates = []
    max_dates = []
    print(lst)
    for item in lst:
        if item is not None:
            print(item)
            if item == 'climatological_averager':
                pass
            else:
                min_dates.append(get_date_range(item)[0])
                max_dates.append(get_date_range(item)[1])
    if max(min_dates) > min(max_dates): 
        #deal with invalid input case
        print(88)
        return [None, None, True, "There is no overlap between your forecast(s) and validation dataset."]
    return [max(min_dates), min(max_dates), False, ""]

map_shorthand = {'acc': 'Anomaly Correlation Coefficient',
'rmse': 'Root Mean Square Error',
'mae': 'Mean Absolute Error',
'bias': 'Forecast Bias'}

def calculate_best(unq, metric):
    res = {}
    if len(unq.keys()) ==  1:
        return [html.Span('Cannot compute best forecast since you are only validating one forecast.')]
    else:
        for key in unq.keys():
            res[key] = abs(mean(unq[key][metric]))
        print(res)
        if metric in ['rmse', 'mae', 'bias']:
            return [html.Span(f'For the {map_shorthand[metric]} Skill Score, the best forecast is '), html.Strong(f"{dash_helper.readable_name(min(res, key=res.get))}.")]
        if metric in ['acc']:
            return [html.Span(f'For the {map_shorthand[metric]} Skill Score, the best forecast is '), html.Strong(f"{dash_helper.readable_name(max(res, key=res.get))}.")]

def calculate_threshold(unq, metric, threshold):
    percent = []
    threshold = float(threshold)
    for key in unq.keys():
        for score in unq[key][metric]:
            percent.append(score)
    thresh = []
    for score in percent:
        if metric in ['rmse', 'mae']:
            if score > threshold:
                thresh.append(score)
        if metric in ['bias']:
            if threshold < 0:
                if score < threshold:
                    thresh.append(score)
            elif threshold >= 0:
                if score > threshold:
                    thresh.append(score)
        if metric in ['acc']:
            if score < threshold:
                thresh.append(score)
    result = np.round(100*len(thresh)/len(percent), 3)
    return str(result)

def ACC(fc,obs,cl):
    top = np.mean((np.array(fc)-np.array(cl))*(np.array(obs)-np.array(cl)))
    bottom = np.sqrt(np.mean((np.array(fc)-np.array(cl))**2)*np.mean((np.array(obs)-np.array(cl))**2))
    if top == 0.0 and bottom == 0.0:
        return 'ACC can not be computed since no difference is detected between forecast and climatological average.'
    else:
        ACC = np.round(top/bottom, 3)
        return ACC

def get_acc(df, df2, forecast_date):
    #historic first, forecast second
    df3 = climatological_averager(40, -120, 'era5_land_wind_u-hourly', forecast_date, fetch_data=False, optional_frame=df)
    df.index = pd.to_datetime(df.index, utc=True)
    df2.index = pd.to_datetime(df2.index, utc=True)
    df3.index = pd.to_datetime(df3.index, utc=True)
    frame = pd.concat([df, df2, df3], axis=1)
    frame.columns = ['Value', 'Value', 'Value']
    frame = frame.dropna(subset=['Value'])
    frame.columns = ['Value', 'ValueF', 'ValueC']
    return ACC(frame['ValueF'], frame['Value'], frame['ValueC'])
    

def accuracy(y1,y2):
    
    accuracy_df=pd.DataFrame()
    
    rms_error = np.round(rmse(y1, y2),1)

    ma_error = np.round(np.mean(np.abs((np.array(y1) - np.array(y2)))))
           
    accuracy_df=accuracy_df.append({"RMSE":rms_error, "MAE": ma_error}, ignore_index=True)
    
    return accuracy_df

#fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

app.layout = html.Div(

    children=[

    dcc.Store(id='memory'),       

    html.H1(children='Forecast Validator'),

html.Div([

    html.Div([
        html.H4(children='Welcome to Forecast Validator!'),
        
        html.P('Fill out the Validation Parameter Inputs to get tailored forecast performance metrics. Your skill scores can be exported as CSV for further analysis, and our Key Insights provide a summary of your validation.'),
        html.H2('Validation Parameter Input'),
    ]),

    html.Div(children=[

        dl.Map([dl.TileLayer(), dl.LayerGroup(id="layer"), *polylines],zoom=6, center=(40.786, -73.962),
            id="map", style={'width': '55vw', 'height': '60vh', 'margin': "10px", }),
            
        html.Div(children=[

        html.H5(children='Forecast selection:'),

        dcc.Dropdown(
            id='forecast',
            options=[
                {'label': 'Climatological Averager', 'value': 'climatological_averager'},
                {'label': 'Global Forecast System 10m East/West Wind', 'value': 'gfs_10m_wind_u-hourly'},
                {'label': 'Global Forecast System 10m North/South Wind', 'value': 'gfs_10m_wind_v-hourly'},
                {'label': 'Global Forecast System Maximum Temperature', 'value': 'gfs_tmax-hourly'},
                {'label': 'Global Forecast System Minimum Temperature',  'value': 'gfs_tmin-hourly'},
                {'label': 'European Centre for Medium-Range Weather Forecasts 2m Temperature', 'value': 'ecmwf_forecasts_temp2m-three_hourly'},
                {'label': 'European Centre for Medium-Range Weather Forecasts 10m East/West Wind', 'value': 'ecmwf_forecasts_windu10m-three_hourly'},
                {'label': 'European Centre for Medium-Range Weather Forecasts 10m North/South Wind', 'value': 'ecmwf_forecasts_windv10m-three_hourly'}                
                #{'label': 'Global Forecast System 10m Full Vector Wind', 'value': 'gfs_10m_wind_full_vector-hourly'},
                    ],
    placeholder='Select a forecast to validate',
    value=None,
    style={'width': '36vw'},
    multi=True,
        ),

        html.H5(children='Validation dataset selection:'),

        dcc.Dropdown(
            id='dataset',
            options=[
                    ],
    placeholder='Select a dataset to validate against',
    value=None,
    style={'width': '36vw'}
        ),

        html.H5(children='Forecast start date selection:'),

        dcc.DatePickerRange(
            id='forecast_date_range',
            #month_format='X',
            #placeholder='X',
            start_date=None,
            end_date=None,
        ),

        html.Div(
        id='date_range_error',
        children=""),

        html.H5(children='Limit forecast length (optional - in days):'),


        html.P('Example: passing 1 here will only validate the first day of a 15 day forecast.'),

        dcc.Input(
            id='end_date',
            #month_format='X',
            placeholder='Limit forecast length (days)',
            value=None,
        ),        

        
        html.H5(children='Location selection:'),

        html.Div(
        id='location',
        children='''
        No location selected. You must select a location to validate.
                '''),

        html.Button(id='clear_markers', n_clicks=0, children='Clear all markers', style={"font-size": "16px", "padding": "6px 16px", "border-radius": "8px"}),        

        ],

            style={#"background-color": "red",
            "display": "box",
            "justify-content": "start",
            "align-items": "center",
            "margin-top": "16px",
            "margin-left": "8px",

            })
            
        ],

 

    style={#"background-color": "blue",
            "display": "flex",
            "justify-content": "start",
            "align-items": "top",
            "margin": "0px"
            },

        ),

    html.Button(id='button', n_clicks=0, children='Run validation', style={"font-size": "16px", "padding": "6px 16px", "border-radius": "8px"}),

    html.Button(id="cancel_button_id", children='Cancel validation', style={"font-size": "16px", "padding": "6px 16px", "border-radius": "8px"}),

    dbc.Container(dbc.Row(dbc.Col(id='fs_spinner', children=None))),

    html.H2('Validation Results'),
    html.H5('Select your Skill Score Metric'),
    dcc.Tabs(id="tabs-example-graph", value='acc_graph', children=[
        dcc.Tab(label='Anomaly Correlation Coefficient', value='acc_graph'),
        dcc.Tab(label='Root Mean Square Error', value='rmse_graph'),
        dcc.Tab(label='Mean Absolute Error', value='mae_graph'),
        dcc.Tab(label='Forecast Bias', value='bias_graph'),
    ]),
    html.Div(id='tabs-content-example-graph'),


    html.H3('Validation Data Table'),

    html.Div(
        id='text',
        children='''
        The results of your validation will appear here.
                '''),

    html.H3(children='Methodology'),

    html.P('For each forecast which is validated, skillscores are derived for four metrics: Anomaly Correlation Coefficient, Root Mean Square Error, Mean Absolute Error, and Forecast Bias.'),

    html.A('Metric info and implementation source', href='https://metclim.ucd.ie/wp-content/uploads/2017/07/DeterministicSkillScore.pdf', target='_blank'),

    html.H5('Anomaly Correlation Coefficient'),

    html.P('Correlations between forecasts and observations may have too high correlations due to seasonal variations therefore the anomaly correlation coefficient (ACC) is used. It removes the climate average from both forecast and observations and verifies the anomalies. Increasing numerical values indicate increasing “success”. An ACC=60% corresponds to the range up to which there is synoptic skill for the largest weather patterns. An ACC= 50% corresponds to forecasts for which the error is the same as for a forecast based on a climatological average.'),

    html.H5('Root Mean Square Error'),

    html.P('The most common accuracy measure is the root mean square error (RMSE) which is a measure of the distance between the forecast and the observation. Lower values of RMSE are better. As the square-root of the MSE is computed, the value is represented in the original physical unit, making it easier to relate to a forecast value. The RMSE penalises large errors more than the non-quadratic MAE and therefore takes higher numerical values.'),

    html.H5('Mean Absolute Error'),

    html.P('The mean absolute error (MAE) sums up the absolute error of each forecast. It is used to determine the overall minimum difference in error values or to find the proportional weighting of errors. It is a linear absolute error measure.'),

    html.H5('Forecast Bias'),    

    html.P('The bias shows if a model overestimates or underestimates a forecast. It’s an average of all single error values. Bias is also known as Mean Error (ME). Positive and negative errors cancel each other out in this score, therefore it provides only an average summary of whether the system overestimates or underestimates, implying a systematic error. The ideal value is zero.  Bias is not a measure of the forecasting quality, but a low bias is desirable and related to a low error. A forecast which overestimates will yield a positive bias and a forecast which underestimates will yield a negative bias.'),

        ])

])

'''
    html.Div([
                html.H2('Powered by:'),
                
                html.Div([
                    html.Img(src=app.get_asset_url('dclimate.jpg'), style={'max-width': '24vw', 'height': '20vh'}),
                ],  style={'width': '24vw', "display": "inline-block", "justify-content": "start", "align-items": "top", "margin": "0px", "vertical-align": "top"},),
                html.Div([
                    html.Img(src=app.get_asset_url('ipfs2.png'), style={'width': '24vw'}),
                ],  style={'width': '24vw', "display": "inline-block", "justify-content": "start", "align-items": "top", "margin": "0px", "vertical-align": "top"},),
                html.Div([
                    html.Img(src=app.get_asset_url('chainlink2.png'), style={'width': '24vw'}),
                ],  style={'width': '24vw', "display": "inline-block", "justify-content": "start", "align-items": "top", "margin": "0px", "vertical-align": "top"},),
                html.Div([
                    html.Img(src=app.get_asset_url('plotly.png'), style={'width': '24vw'}),
                ],  style={'width': '24vw', "display": "inline-block", "justify-content": "start", "align-items": "top", "margin": "0px", "vertical-align": "top"},),
            ], style={"display": "inline-block"})
'''
'''
#callback to update slider max/min 
@app.callback(
    [Output('threshold_slider', 'marks'),
    Output('threshold_slider', 'min'),
    Output('threshold_slider', 'max'),
    Output('threshold_slider', 'value')],
    [Input('tabs-example-graph', 'value'),
    Input('memory', 'data')]
)
def update_slider(tab, data):
    if data is not None:
        if tab == 'acc_graph':
            ind = 0
        elif tab == 'rmse_graph':
            ind = 1
        elif tab == 'mae_graph':
            ind = 2
        elif tab == 'bias_graph':
            ind = 3
        data = json.loads(data)
        skillscores = []
        for key in data.keys():
            skillscores.append(data[key][ind])
        mn = np.round(round_decimals_down(min(skillscores)) - 0.01, 2)
        mx = np.round(round_decimals_up(max(skillscores)) + 0.01, 2)

        if tab == 'acc_graph':
            mx = 1

        return [{mn: {'label': f'{mn} (min)'}, mx: {'label': f'{mx} (max)'}}, mn, mx, mean([mn, mx])]
    else:
        return [None, None, None, None]
'''
@app.callback(Output('tabs-content-example-graph', 'children'),
              [Input('tabs-example-graph', 'value'), Input('memory', 'data')])
def render_content(tab, data):

    map_metric = {'acc_graph': 0, 'rmse_graph': 1, 'mae_graph': 2, 'bias_graph': 3}
    list_metrics = ['acc', 'rmse', 'mae', 'bias']
    mapped = map_metric[tab]
    
    if data is not None:
        data = json.loads(data)
        ids = {}
        for key in data.keys():
            bing = key.split('|')
            temp = bing.pop(2)
            bing = '|'.join(bing)
            if bing not in ids.keys():
                ids[bing]= [[temp],[data[key][mapped]]]
            else: 
                ids[bing][0].append(temp)
                ids[bing][1].append(data[key][mapped])
        
        print(88)
        print(ids)

        forecasts = [] 

        x = ids[next(iter(ids))][0]
        yy = []
        '''
        threshold_y = [] 
        for i in range(len(x)):
            threshold_y.append(slider)
        yy.append(threshold_y)
        '''
        for key in ids.keys():
            yy.append(ids[key][1])
            fc = key.split('|')[0]
            if fc not in forecasts:
                forecasts.append(fc)
        
        performance = []
        for fc in forecasts:
            temp = {}
            avg = []
            for key in ids.keys():
                if fc in key:
                    avg.append(mean(ids[key][1]))
            temp['Forecast'] = dash_helper.readable_name(fc)
            temp['μ-Score'] = np.round(mean(avg), 3)
            performance.append(temp)
        print(performance)
        if mapped == 0: 
            sortd = sorted(performance, key=lambda x: x['μ-Score'], reverse=True)
        elif mapped in [1, 2]:
            sortd = sorted(performance, key=lambda x: x['μ-Score'], reverse=False)
        elif mapped == 3:
            sortd = sorted(performance, key=lambda x: abs(x['μ-Score']), reverse=False)
        counter = 0
        newdictolist = []
        for dicto in sortd:
            ranking = sortd.index(dicto) + 1
            for dicto2 in performance: 
                if dicto2 == dicto:
                    newdicto = {}
                    dicto2['Ranking'] = ranking
                    newdicto['Ranking'] = dicto2['Ranking']
                    newdicto['Forecast'] = dicto2['Forecast']
                    newdicto['μ-Score'] = dicto2['μ-Score']
                    newdictolist.append(newdicto)
        
        performance = newdictolist


        
        print(x)
        print(yy)
        fig = px.line(x=x, y=yy, labels={"x": "Forecast Start Date", "value": "Metric Skill Score"})
        fig.update_layout(title_text='', title_x=0.5)
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))


        
        #fig.data[0].name = 'threshold value'
        #fig.data[0].hovertemplate = 'threshold value'
        counter = 0
        for key in ids.keys():
            key = key.split('|')
            key[1] = key[1][1:-1].split(',')
            key[1][0] = str(np.round(float(key[1][0]), 3))
            key[1][1] = str(np.round(float(key[1][1]), 3))
            key[1] = '[' + ','.join(key[1]) + ']'
            key = '|'.join(key)
            fig.data[counter].name = key 
            fig.data[counter].hovertemplate = key
            counter = counter + 1



        unq = {}
        for key in data.keys(): 
            if key.split('|')[0] not in unq.keys():
                unq[key.split('|')[0]] = {'acc': [data[key][0]],
                                  'rmse': [data[key][1]],
                                  'mae': [data[key][2]],
                                  'bias': [data[key][3]],
                                 }
            else:
                unq[key.split('|')[0]]['acc'].append(data[key][0])
                unq[key.split('|')[0]]['rmse'].append(data[key][1])
                unq[key.split('|')[0]]['mae'].append(data[key][2])
                unq[key.split('|')[0]]['bias'].append(data[key][3])

        best = calculate_best(unq, list_metrics[mapped])
        #threshold = calculate_threshold(unq, list_metrics[mapped], slider)
        

    if tab == 'acc_graph':
        if data is None:
            return html.P("Once you have completed a validation, your Skill Scores will appear here!")
        else:
            dt_list = [{'Forecast': 'gfs', 'μ-Score': '2', 'Ranking': '1'}, {'Forecast': 'ecmwf', 'μ-Score': '3', 'Ranking': '2'}]
            return html.Div([
                html.P('Tip: The Anomaly Correlation Coefficient is best for figuring out your forecasts ability to predict deviations from the norm. The score will be between -1 and 1 and the closer to 1, the better!'),
                
                html.Div([
                    html.H3('Key Insights'),
                    html.P('Tip: Key Insights (highlighted in bold) provide actionable summaries of your forecast validation, which we aim to bring on-chain!'),
                    html.H4('Best Forecast:'),
                    html.P(children = best),
                    html.H4('Forecast Ranking:'),
                    dash_table.DataTable(performance),
                    #html.P(children = [
                    #    html.Span('For the Anomalaly Correlation Coefficient Skill Score, the best forecast is'),
                    #    html.Strong(f'{best}%.'),
                    #]),

                ],  style={'width': '46vw', "display": "inline-block", "justify-content": "start", "align-items": "top", "margin": "0px", "vertical-align": "top"},),
 
                html.Div([
                    html.H3('Skill Score Graph'),
                    dcc.Graph(figure = fig),
                ], style={'width': '52vw', "display": "inline-block", "justify-content": "start", "align-items": "top", "margin": "0px"}),

            ], style={"display": "inline-block"})
    elif tab == 'rmse_graph':
        if data is None:
            return html.P("Once you have completed a validation, your Skill Scores will appear here!")
        else:
            return html.Div([
                html.P('Tip: The Root Mean Square Error will be in the units of your weather variable, and penalizes errors more the larger they are. The lower the better!'),
                
                html.Div([
                    html.H3('Key Insights'),
                    html.P('Tip: Key Insights (highlighted in bold) provide actionable summaries of your forecast validation, which we aim to bring on-chain!'),
                    html.H4('Best Forecast:'),
                    html.P(children = best),
                    html.H4('Forecast Ranking:'),
                    dash_table.DataTable(performance),
                    #html.P(children = [
                    #    html.Span('For the Anomalaly Correlation Coefficient Skill Score, the best forecast is'),
                    #    html.Strong(f'{best}%.'),
                    #]),

                ],  style={'width': '46vw', "display": "inline-block", "justify-content": "start", "align-items": "top", "margin": "0px", "vertical-align": "top"},),
 
                html.Div([
                    html.H3('Skill Score Graph'),
                    dcc.Graph(figure = fig),
                ], style={'width': '52vw', "display": "inline-block", "justify-content": "start", "align-items": "top", "margin": "0px"}),

            ], style={"display": "inline-block"})
    elif tab == 'mae_graph':
        if data is None:
            return html.P("Once you have completed a validation, your Skill Scores will appear here!")
        else:
            return html.Div([
                html.P('Tip: The Mean Absolute Error is the most simple skill score and tells you the average absolute difference between forecast and observed values. The lower the better!'),
                
                html.Div([
                    html.H3('Key Insights'),
                    html.P('Tip: Key Insights (highlighted in bold) provide actionable summaries of your forecast validation, which we aim to bring on-chain!'),
                    html.H4('Best Forecast:'),
                    html.P(children = best),
                    html.H4('Forecast Ranking:'),
                    dash_table.DataTable(performance),
                    #html.P(children = [
                    #    html.Span('For the Anomalaly Correlation Coefficient Skill Score, the best forecast is'),
                    #    html.Strong(f'{best}%.'),
                    #]),

                ],  style={'width': '46vw', "display": "inline-block", "justify-content": "start", "align-items": "top", "margin": "0px", "vertical-align": "top"},),
 
                html.Div([
                    html.H3('Skill Score Graph'),
                    dcc.Graph(figure = fig),
                ], style={'width': '52vw', "display": "inline-block", "justify-content": "start", "align-items": "top", "margin": "0px"}),

            ], style={"display": "inline-block"})
    elif tab == 'bias_graph':
        if data is None:
            return html.P("Once you have completed a validation, your Skill Scores will appear here!")
        else:
            return html.Div([
                html.P('Tip: The Forecast Bias tells you whether your forecast overpredicts or underpredicts. A positive value indicates overestimating, and a negative value indicates underestimating. The lower absolute value, the better!'),
                
                html.Div([
                    html.H3('Key Insights'),
                    html.P('Tip: Key Insights (highlighted in bold) provide actionable summaries of your forecast validation, which we aim to bring on-chain!'),
                    html.H4('Best Forecast:'),
                    html.P(children = best),
                    html.H4('Forecast Ranking:'),
                    dash_table.DataTable(performance),
                    #html.P(children = [
                    #    html.Span('For the Anomalaly Correlation Coefficient Skill Score, the best forecast is'),
                    #    html.Strong(f'{best}%.'),
                    #]),

                ],  style={'width': '46vw', "display": "inline-block", "justify-content": "start", "align-items": "top", "margin": "0px", "vertical-align": "top"},),
 
                html.Div([
                    html.H3('Skill Score Graph'),
                    dcc.Graph(figure = fig),
                ], style={'width': '52vw', "display": "inline-block", "justify-content": "start", "align-items": "top", "margin": "0px"}),

            ], style={"display": "inline-block"})

@app.callback(
   [
       Output(component_id='dataset', component_property='options'),
   ],
   [Input(component_id='forecast', component_property='value')])


def update_valid_datasets(forecast):

    # vv todo fix this, this is sloppy
    if forecast not in [None]:
        for fc in forecast:
            valid = algo.VALIDATION_PER_FORECAST[fc]
            result = dash_helper.to_dash_list_dict(valid)
        return [result]

@app.long_callback(
    [
        Output(component_id='text', component_property='children'),
        Output('memory', 'data'),
    ],

    [
        Input('button', 'n_clicks'),
        State(component_id='forecast', component_property='value'),
        State(component_id='dataset', component_property='value'),
        State(component_id='map', component_property='click_lat_lng'),
        State(component_id='layer', component_property='children'),
        State(component_id='end_date', component_property='value'),
        State(component_id='forecast_date_range', component_property='start_date'),  
        State(component_id='forecast_date_range', component_property='end_date'),                          
    ],

    running=[
        (Output("button", "disabled"), True, False),
        (Output("cancel_button_id", "disabled"), False, True),
        (Output('fs_spinner', 'children'), dbc.Spinner(size='md', type="grow"), None)
    ],
    cancel=[Input("cancel_button_id", "n_clicks")],
)
def update(n_clicks, forecast, dataset, latlong, map, end_date, forecast_date_range_start, forecast_date_range_end):


    if n_clicks > 0:

        #determine lat/long input 
        #to do: generate unrounded int list
        latlongz = [[latlong[0], latlong[1]]]
        if type(map) == list:
            latlongz = []
            for i in range(len(map)):
                latlongz.append([map[i]['props']['position'][0], map[i]['props']['position'][1]])
        #to do2: generate rounded string list
        latlongz_rounded = [] 
        for pair in latlongz:
            latlongz_rounded.append(
                [np.round(pair[0],3), np.round(pair[1],3)]
            )
        print(latlongz)

        datez_list = [] 
        startdt = datetime.datetime.strptime(forecast_date_range_start, '%Y-%m-%d')
        enddt = datetime.datetime.strptime(forecast_date_range_end, '%Y-%m-%d')
        delta = enddt - startdt   # returns timedelta

        for i in range(delta.days + 1):
            day = startdt + datetime.timedelta(days=i)
            datez_list.append(str(day)[0:10])

        query = ValidationQuery(datez_list, latlongz, forecast, dataset, length_limit=end_date)
        metrics = {}
        for key in query.identifiers.keys():
            metrics[key] = query.identifiers[key]['metrics']

        dt_list = []
        for key in metrics.keys():
            tempdic = {}
            #tempdic['Validation Identifier'] = query.identifiers[key]['rounded_identifier']
            tempdic['Date'] = query.identifiers[key]['date']
            tempdic['Lat'] = np.round(query.identifiers[key]['lat'], 3)
            tempdic['Long'] = np.round(query.identifiers[key]['long'], 3)
            tempdic['Forecast'] = query.identifiers[key]['forecast']
            tempdic['Validation Dataset'] = query.identifiers[key]['dataset']
            tempdic['Anomaly Correlation Coefficient'] = metrics[key][0]
            tempdic['Root Mean Square Error'] = metrics[key][1]
            tempdic['Mean Absolute Error'] = metrics[key][2]
            tempdic['Forecast Bias'] = metrics[key][3]
            dt_list.append(tempdic)

        date_list = []
        for idtfr in metrics.keys():
            if query.identifiers[idtfr]['date'] not in date_list:
                date_list.append(query.identifiers[idtfr]['date'])
        yy = []

        banana = [] 
        for i in range(4):
            for identifier in query.identifiers.keys():
                banana.append(query.identifiers[identifier]['metrics'][i])            
        yy = [banana[x:x+len(date_list)] for x in range(0, len(banana),len(date_list))]


        print(banana)
        print(len(banana))
        print(yy)

        fig = px.line(x=date_list, y=yy, labels={"x": "Forecast Start Date", "value": "Metric Skill Score"})
        fig.update_layout(title_text='Forecast Metric Validation', title_x=0.5)
        naming_list = []
        unique = []
        for unq in query.identifiers.keys():
            if date_list[0] in unq:
                rnd_list = query.identifiers[unq]['rounded_identifier'].split('|')
                rnd_list.pop(2)
                rnd_list = '|'.join(rnd_list)
                unique.append(rnd_list)
        for mtrc in [' ACC', ' RMSE', ' MAE', ' BIAS']:
            for unq in unique:
                naming_list.append(unq+mtrc)
        for i in range(len(naming_list)):
            fig.data[i].name = naming_list[i]

        print(77)
        print(metrics)
        data = json.dumps(metrics) 

        return [dash_table.DataTable(dt_list, export_format="csv",page_size=50, style_table={'max-height': '300px', 'overflowY': 'auto'}), data]
    else:
        return ["The results of your validation will appear here.", None]

@app.callback([Output("layer", "children"), Output("location", "children")], [Input("map", "click_lat_lng"), Input("layer", "children"), Input("clear_markers", "n_clicks"), Input("map", "dbl_click_lat_lng")])
def map_click(click_lat_lng, current, n_clicks, dbl_click_lat_lng):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]

    if 'clear_markers' in changed_id:
        return  [None, "No location selected. You must select a location to validate."]
    
    else:
        if current is None:
                return [dl.Marker(position=click_lat_lng, children=dl.Tooltip("({:.3f}, {:.3f})".format(*click_lat_lng))), '[' + str(np.round(click_lat_lng[0], 3)) + ',' + str(np.round(click_lat_lng[1], 3))+']']
        else:
            new = dl.Marker(position=click_lat_lng, children=dl.Tooltip("({:.3f}, {:.3f})".format(*click_lat_lng)))
            #old = dl.Marker(position=current[0]['props']['position'], children=current[0]['props']['children'])
            #print(current)
            results = []
            latzlongz = []

            if type(current) == dict:
                print(current)
                old = dl.Marker(position=current['props']['position'], children=current['props']['children'])
                #print(old)
                results.append(old)
                latzlongz.append(current['props']['position'])
            else:  
                for i in range(len(current)):
                    print(current)
                    old = dl.Marker(position=current[i]['props']['position'], children=current[i]['props']['children'])
                    #print(old)
                    results.append(old)
                    latzlongz.append(current[i]['props']['position'])
                    #print(results)

            results.append(new)
            latzlongz.append([click_lat_lng[0], click_lat_lng[1]])

            for i in range(len(latzlongz)):
                latzlongz[i] = [np.round(latzlongz[i][0], 3), np.round(latzlongz[i][1], 3)]

            return [results, str(latzlongz)]

'''
# output the stored clicks in the table cell.
@app.long_callback(
    [
        Output('best', 'children'),

    ],

    [
        Input('derived_best_button', 'n_clicks'),
        State('memory', 'data'),
        State('best_metric', 'value')                      
    ],
)
def update_best(n_clicks, data, best_metric):
    if n_clicks not in  [None]:

        processed = json.loads(data)
        unq = {}
        for key in processed.keys(): 
            if key.split('|')[0] not in unq.keys():
                unq[key.split('|')[0]] = {'acc': [processed[key][0]],
                                  'rmse': [processed[key][1]],
                                  'mae': [processed[key][2]],
                                  'bias': [processed[key][3]],
                                 }
            else:
                unq[key.split('|')[0]]['acc'].append(processed[key][0])
                unq[key.split('|')[0]]['rmse'].append(processed[key][1])
                unq[key.split('|')[0]]['mae'].append(processed[key][2])
                unq[key.split('|')[0]]['bias'].append(processed[key][3])
        
            
        return [html.P(children=calculate_best(unq, best_metric))]
    else: 
        return [html.P(children='The results of your best forecast validation will appear here.')]

@app.long_callback(
    [
        Output('threshold', 'children'),
    ],

    [
        Input('derived_threshold_button', 'n_clicks'),
        State('memory', 'data'),
        State('threshold_metric', 'value'),
        State('threshold_value', 'value')                   
    ],
)

def update_threshold(n_clicks, data, threshold_metric, threshold_value):
    if n_clicks not in  [None]:

        processed = json.loads(data)
        unq = {}
        for key in processed.keys(): 
            if key.split('|')[0] not in unq.keys():
                unq[key.split('|')[0]] = {'acc': [processed[key][0]],
                                  'rmse': [processed[key][1]],
                                  'mae': [processed[key][2]],
                                  'bias': [processed[key][3]],
                                 }
            else:
                unq[key.split('|')[0]]['acc'].append(processed[key][0])
                unq[key.split('|')[0]]['rmse'].append(processed[key][1])
                unq[key.split('|')[0]]['mae'].append(processed[key][2])
                unq[key.split('|')[0]]['bias'].append(processed[key][3])
        
            
        return [html.P(children=calculate_threshold(unq, threshold_metric, threshold_value))]
    else: 
        return [html.P(children='The results of your threshold validation will appear here.')]

'''
'''
@app.long_callback(
    [
        Output('slider_div', 'style')
    ],

    [
        Input('memory', 'data'),                
    ],
)
def update_valid_metric_buttons(data):
    print(type(data))
    if data is None:
        return [{'display': 'none'}]
    else:
        return [{'width': '60vw', 'margin-left': 'auto', 'margin-right': 'none'}]
'''
@app.long_callback(
    [
        Output('forecast_date_range', 'min_date_allowed'),
        Output('forecast_date_range', 'max_date_allowed'),
        Output('forecast_date_range', 'disabled'),
        Output('date_range_error', 'children')
    ],

    [
        Input('forecast', 'value'),
        Input('dataset', 'value')                
    ],
)
def update_valid_metric_buttons(forecast, dataset):
    print(forecast, dataset)
    if forecast is not None:
        if dataset is not None:
            print(forecast)
            print(dataset)
            forecast.append(dataset)

            return date_range_extreme(forecast)
        else: 
            return [None, None, False, None]

    else:
        return [None, None, False, None]

if __name__ == '__main__':
    app.run_server(debug=True, port=8052)

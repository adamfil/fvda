import requests
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime, timedelta
import pytz


def climatological_averager(lat, long, dataset, forecast_date, forecast_length=720, fetch_data=True, optional_frame=None, return_both=False, end_date=None):
    #ensure forecast_date is in yyyy-mm-dd format 
    yr = str(forecast_date)[0:4]
    dat = str(forecast_date)[5:]

    my_token = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjI1MzQwMjMwMDc5OSwiaWF0IjoxNjMwNTMwMTgwLCJzdWIiOiJhMDIzYjUwYi0wOGQ2LTQwY2QtODNiMS1iMTExZDA2Mzk1MmEifQ.qHy4B0GK22CkYOTO8gsxh0YzE8oLMMa6My8TvhwhxMk'

    if fetch_data == True:
        my_url = 'https://api.dclimate.net/apiv3/grid-history/' + dataset + '/' + str(lat) + '_' + str(long)
        head = {"Authorization": my_token}
        r = requests.get(my_url, headers=head)
        data = r.json()["data"]
        index = pd.to_datetime(list(data.keys()))
        values = [float(s.split()[0]) if s else None for s in data.values()]
        series = pd.Series(values, index=index)
        df = series.to_frame(name='ValueF')
        if return_both==True:
            both = series.to_frame(name='Value')

    else:
        df = optional_frame

    df = series.to_frame(name='ValueF')
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.astype(str).str.contains('02-29')]
    df = df.groupby([df.index.month, df.index.day, df.index.hour]).mean()
    df.index = pd.to_datetime(df.index.get_level_values(0).astype(str) + '-' +
               df.index.get_level_values(1).astype(str) + '-' +
               df.index.get_level_values(2).astype(str),
               format='%m-%d-%H')
    df1 = df 
    df2 = df1.copy()
    df1.index = df1.index.map(lambda t: t.replace(year=int(yr)))
    df2.index = df2.index.map(lambda t: t.replace(year=int(yr) + 1))
    df = pd.concat([df1, df2]) 
    
    ind = df.index.astype(str).str.contains(dat)
    itemindex = np.where(ind==True)
    start = itemindex[0][0]
    if end_date in [None]:
        end = start + forecast_length
        df = df[start:end]
    else:
        end_year = int(end_date[0:4])
        end_month = int(end_date[5:7])
        end_day = int(end_date[8:10])
        end = datetime(end_year, end_month, end_day, 0, 0, 0, tzinfo=pytz.UTC)
        start_year = int(forecast_date[0:4])
        start_month = int(forecast_date[5:7])
        start_day = int(forecast_date[8:10])
        start = datetime(start_year, start_month, start_day, 0, 0, 0, tzinfo=pytz.UTC)  
        print(start)
        print(end)
        df = df[start:end]
    if return_both==False:
        return df
    else:
        return [both, df]

def new_climatological_averager(df, year):
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.astype(str).str.contains('02-29')]
    df = df.groupby([df.index.month, df.index.day, df.index.hour]).mean()
    df.index = pd.to_datetime(df.index.get_level_values(0).astype(str) + '-' +
               df.index.get_level_values(1).astype(str) + '-' +
               df.index.get_level_values(2).astype(str),
               format='%m-%d-%H')
    df1 = df 
    df2 = df1.copy()
    df1.index = df1.index.map(lambda t: t.replace(year=int(year)))
    df2.index = df2.index.map(lambda t: t.replace(year=int(year) + 1))
    df = pd.concat([df1, df2]) 
    df = df.rename(columns={'Value': 'ValueC'})
    
    return df
'''
## in order to fix 0.5 bug, i should simply slice the climatological dataframe
def new_climatological_forecaster(df, forecast_date, length):
    if length in [None, '']:
        #if no length is povided, climatological forecaster defualts to a 15-day forecast from the start date
        length = 15
    yr = str(forecast_date)[0:4]
    dat = str(forecast_date)[5:]
    #df = series.to_frame(name='ValueF')
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.astype(str).str.contains('02-29')]
    df = df.groupby([df.index.month, df.index.day, df.index.hour]).mean()
    df.index = pd.to_datetime(df.index.get_level_values(0).astype(str) + '-' +
               df.index.get_level_values(1).astype(str) + '-' +
               df.index.get_level_values(2).astype(str),
               format='%m-%d-%H')
    df1 = df 
    df2 = df1.copy()
    df1.index = df1.index.map(lambda t: t.replace(year=int(yr)))
    df2.index = df2.index.map(lambda t: t.replace(year=int(yr) + 1))
    df = pd.concat([df1, df2]) 
    
    ind = df.index.astype(str).str.contains(dat)
    itemindex = np.where(ind==True)
    start = itemindex[0][0]
    end = start + int(length)*24
    df = df[start:end]
    df = df.rename(columns={"Value": "ValueF"})
    
    return df
'''
## 0.5 error is occuring because i am using the same climatological frame for 2 diff locations

def new_climatological_forecaster(df, forecast_date, length):
    if length in [None, '']:
        #if no length is povided, climatological forecaster defualts to a 15-day forecast from the start date
        length = 15

    startdt = datetime.strptime(forecast_date, '%Y-%m-%d')

    enddt = startdt + timedelta(days=int(length))  

    df = df[str(startdt):str(enddt)]
    #print(str(startdt), str(enddt))
    df = df.rename(columns={"ValueC": "ValueF"})
    
    return df
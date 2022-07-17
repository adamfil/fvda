#from pydoc import cli
import pandas as pd
import requests 
import numpy as np
from climatological_averager import new_climatological_averager, new_climatological_forecaster, climatological_averager
from statsmodels.tools.eval_measures import rmse
import pytz 
from datetime import datetime, timedelta

MY_TOKEN = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjI1MzQwMjMwMDc5OSwiaWF0IjoxNjMwNTMwMTgwLCJzdWIiOiJhMDIzYjUwYi0wOGQ2LTQwY2QtODNiMS1iMTExZDA2Mzk1MmEifQ.qHy4B0GK22CkYOTO8gsxh0YzE8oLMMa6My8TvhwhxMk'
#define all forecasts which can be validated
VALID_FORECASTS = ['climatological_averager', 'gfs_10m_wind_u-hourly', 'gfs_10m_wind_v-hourly', 'gfs_tmax-hourly', 'gfs_tmin-hourly', 
'ecmwf_forecasts_temp2m-three_hourly', 'ecmwf_forecasts_windu10m-three_hourly', 'ecmwf_forecasts_windv10m-three_hourly']

#for each valid forecast, define which metrics will be measured
METRICS_PER_FORECAST = {'climatological_averager': ['rmse', 'mae', 'bias'], 
'gfs_10m_wind_u-hourly': ['rmse', 'mae', 'acc', 'bias'], 
'gfs_10m_wind_v-hourly': ['rmse', 'mae', 'acc', 'bias'],
#'gfs_10m_wind_full_vector-hourly': ['acc', 'rmswve'],
'gfs_tmax-hourly': ['rmse', 'mae', 'acc', 'bias'],
'gfs_tmin-hourly': ['rmse', 'mae', 'acc', 'bias'],
}

#for each valid forecast, define which datasets can be used as validation datasets
VALIDATION_PER_FORECAST = {
    'gfs_10m_wind_u-hourly': ['era5_land_wind_u-hourly', 'era5_wind_100m_u-hourly', 'rtma_wind_u-hourly'],
    'gfs_10m_wind_v-hourly': ['era5_land_wind_v-hourly', 'era5_wind_100m_v-hourly', 'rtma_wind_v-hourly'],
    'gfs_tmax-hourly': ['era5_land_2m_temp-hourly', 'rtma_temp-hourly'],
    'gfs_tmin-hourly': ['era5_land_2m_temp-hourly', 'rtma_temp-hourly'],
    'climatological_averager': ['era5_land_wind_v-hourly', 'era5_wind_100m_v-hourly', 'era5_land_wind_u-hourly', 'era5_wind_100m_u-hourly', 
'era5_land_surface_solar_radiation_downwards-hourly', 'era5_land_2m_temp-hourly', 'era5_land_precip-hourly', 'rtma_wind_u-hourly', 'rtma_wind_v-hourly', 'rtma_temp-hourly'],
    'ecmwf_forecasts_temp2m-three_hourly': ['era5_land_2m_temp-hourly', 'rtma_temp-hourly'],
    'ecmwf_forecasts_windu10m-three_hourly': ['era5_land_wind_u-hourly', 'era5_wind_100m_u-hourly', 'rtma_wind_u-hourly'],
    'ecmwf_forecasts_windv10m-three_hourly': ['era5_land_wind_v-hourly', 'era5_wind_100m_v-hourly', 'rtma_wind_v-hourly'],
    }

MAP_METRICS = {'ACC': 0,
    'RMSE': 1,
    'MAE': 2, 
    'BIAS': 3
}
#define all our metric functions
def ACC(fc,obs,cl):
    top = np.mean((np.array(fc)-np.array(cl))*(np.array(obs)-np.array(cl)))
    bottom = np.sqrt(np.mean((np.array(fc)-np.array(cl))**2)*np.mean((np.array(obs)-np.array(cl))**2))
    if top == 0.0 and bottom == 0.0:
        #change check to if fc array - cl array == 0.0 and return 0.5 (error of climatological avg.)
        return 0.5
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
    
    rms_error = np.round(rmse(y1, y2),2)

    ma_error = np.round(np.mean(np.abs((np.array(y1) - np.array(y2)))),2)
           
    accuracy_df=accuracy_df.append({"RMSE":rms_error, "MAE": ma_error}, ignore_index=True)
    
    return accuracy_df

class ValidationQuery:
    def __init__(self, timeframe, location, forecast, dataset, length_limit=None):

        #assert forecast in VALID_FORECASTS, f'{forecast} is not a valid forecast name'
        #assert dataset in VALIDATION_PER_FORECAST[forecast], f'{dataset} is not a valid validation dataset for {forecast}'  
        #^^ need to add assert checks above, only allow valid inputs past this point
        #only three forms of timeframe are accepted:
        #either '2022-01-01' (single) 
        #or ('2022-01-01', '2022-01-01') (range)
        #or ['2022-01-01', '2021-12-12', '2022-02-02'] (list_)
        ## all of these will be transformed to [] format

        if type(forecast) == str:
            self.forecast = [forecast]
        else:
            self.forecast = forecast

        self.dataset = dataset 
        self.fetched_dataset = False
        self.length_limit = length_limit

        if type(timeframe) == list:
            self.timeframe = timeframe
            self.timeframe_type = 'list'
        elif type(timeframe) == str:
            self.timeframe = [timeframe]
            self.timeframe_type = 'single'
        elif type(timeframe) == tuple:
            ##todo: assign self.timeframe properly by looping through the days 
            timeframe_list = []
            startdt = datetime.strptime(timeframe[0], '%Y-%m-%d')
            enddt = datetime.strptime(timeframe[1], '%Y-%m-%d')
            delta = enddt - startdt   # returns timedelta

            for i in range(delta.days + 1):
                day = startdt + timedelta(days=i)
                timeframe_list.append(str(day)[0:10])            
            self.timeframe = timeframe_list
            self.timeframe_type = 'range'

        #only two forms of location are accepted:
        #either [40.3234, 34.432] (single, gets converted to [[40.3234, 34.432]], which is also a single and also accepted)
        #or [[37.123, 115.234],[36.43, 110.23]] (multiple)
        if type(location[0]) == list:
            self.location = location 
        else: 
            self.location = [location]
        if len(self.location) == 1:
            self.location_type = 'single'
        else:
            self.location_type = 'multiple'
        rounded = [] 
        for latlong in self.location:
            rounded.append([np.round(float(latlong[0]),3), np.round(float(latlong[1]),3)])
        self.location_rounded = rounded

        datasets = {}
        for location in self.location:
            #add condition for custom dataset:
            #custom datasets must be passed in the format:
            #{
                #csv_path 
                #lat column 
                #long column 
                #variable column
            # }
            print("Fetching data for " + self.dataset + ' at ' + str(location))
            my_url = 'https://api.dclimate.net/apiv3/grid-history/' + self.dataset + '/' + str(location[0]) + '_' + str(location[1])
            head = {"Authorization": MY_TOKEN}
            r = requests.get(my_url, headers=head)
            ## should add cond here to catch errors
            data = r.json()["data"]
            index = pd.to_datetime(list(data.keys()))
            values = [float(s.split()[0]) if s else None for s in data.values()]
            series = pd.Series(values, index=index)
            datasets[str(location)] = series.to_frame(name='Value')
        self.datasets = datasets

        climatological = {}
        for key in datasets.keys():
            #to do: need to generalize year input, currently hard coded to only work with year 2021. the way to do this iwll be by taking min max of all forecast dates input
            climatological[key] = new_climatological_averager(datasets[key], year=2021)
        
        self.climatological = climatological

        '''
        climatological = {}
        for key in datasets.keys():
            climatological[key] = new_climatological_averager(datasets[key])
        self.climatological = climatological
        climatological_trimmed = {}
        for i in range(len(self.identifiers)):
            climatological_data = climatological[str(self.lats_and_longs[i])]
            trimmed = climatological_trimmer(climatological_data)
            climatological_trimmed[self.identifiers[i]] = trimmed
        self.climatological_trimmed = climatological_trimmed
        '''
        forecasts = {}
        for latlong in self.location:
            for date in self.timeframe:
                for forecast in self.forecast:
                    if forecast not in ['climatological_averager']:
                        #deal with base case of regular forecast
                        print("Fetching data for " + forecast + ' at ' + str(latlong[0]) + ' ' + str(latlong[1]) + ' on ' + str(date))
                        my_url = 'https://api.dclimate.net/apiv3/forecasts/' + forecast + '/' + str(latlong[0]) + '_' + str(latlong[1]) + '?forecast_date=' + str(date)
                        head = {"Authorization": MY_TOKEN}
                        r = requests.get(my_url, headers=head)
                        #print(r.text)
                        data = r.json()["data"]
                        index = pd.to_datetime(list(data.keys()))
                        values = [float(s.split()[0]) if s else None for s in data.values()]
                        series = pd.Series(values, index=index)
                        if self.length_limit not in [None, '']:
                            end_date = str(datetime.strptime(date, '%Y-%m-%d') + timedelta(days=int(self.length_limit)))[0:11]
                            series = series[:end_date]
                        '''self
                        if self.end_date not in [None]:
                                    end_year = int(self.end_date[0:4])
                                    end_month = int(self.end_date[5:7])
                                    end_day = int(self.end_date[8:10])
                                    end = datetime(end_year, end_month, end_day, 0, 0, 0, tzinfo=pytz.UTC)
                                    series = series[:end]
                        '''
                        data = series.to_frame(name='ValueF')
                    else:
                        #deal with case of climatological averager forecast
                        #data = new_climatological_forecaster(self.datasets[str(latlong)], date, self.length_limit)
                        data = new_climatological_forecaster(self.climatological[str(latlong)], date, self.length_limit)
                    forecasts[f"{forecast}|{latlong}|{date}"] = data
        self.forecasts = forecasts

        identifiers = {}
        for forecast in self.forecast:
            for latlong in self.location:
                for date in self.timeframe:
                    ##vv below todo is writing a rounder faction so latlong are rounded in idtfr and calling readable name transformer function on forecast/dataset
                    if self.length_limit not in [None, '']:
                        idtfr = f'{forecast}|{latlong}|{date}|{self.dataset}|limit={self.length_limit} days'
                        rounded_idtfr = f'{forecast}|[{np.round(latlong[0], 3)},{np.round(latlong[1], 3)}]|{date}|{self.dataset}|limit={self.length_limit} days'
                    else:
                        idtfr = f'{forecast}|{latlong}|{date}|{self.dataset}'
                        rounded_idtfr = f'{forecast}|[{np.round(latlong[0], 3)},{np.round(latlong[1], 3)}]|{date}|{self.dataset}'
                    identifiers[idtfr] = {
                            'forecast': forecast,
                            'location': latlong,
                            'lat': latlong[0],
                            'long': latlong[1],
                            'date': date,
                            'dataset': self.dataset,
                            #'forecast_data': 'POINT THIS TO THE KEY OF THE FORECAST',
                            'forecast_data': self.forecasts[f"{forecast}|{str(latlong)}|{date}"],
                            #'validation_data': 'POIT THIS TO THE KEY OF THE DATASET',
                            'validation_data': self.datasets[str(latlong)],
                            'climatological_data': self.climatological[str(latlong)],
                            #'climatological_data': self.climatological[str(latlong)],
                            'metrics': None,
                            'rounded_identifier': rounded_idtfr,
                            }             
        self.identifiers = identifiers

        for identifier in self.identifiers:
            
            validation_frame = self.identifiers[identifier]['validation_data']
            forecast_frame = self.identifiers[identifier]['forecast_data']
            climatological_frame = self.identifiers[identifier]['climatological_data']

            validation_frame.index = pd.to_datetime(validation_frame.index, utc=True)
            forecast_frame.index = pd.to_datetime(forecast_frame.index, utc=True)
            climatological_frame.index = pd.to_datetime(climatological_frame.index, utc=True)
            frame = pd.concat([validation_frame, forecast_frame, climatological_frame], axis=1)
            frame.columns = ['plchldr', 'plchldr', 'plchldr']
            frame = frame.dropna(subset=['plchldr'])
            frame.columns = ['Value', 'ValueF', 'ValueC']
            acc = ACC(frame['ValueF'], frame['Value'], frame['ValueC']) 

            #compute diff between forecast and validation for bias
            frame['Diff'] = frame['ValueF'].sub(frame['Value'], axis = 0)
            
            rmse = accuracy(frame['Value'], frame['ValueF'])['RMSE'][0]
            mae = accuracy(frame['Value'], frame['ValueF'])['MAE'][0]
            bias = np.round(frame['Diff'].mean(), 3)


            metrics = [acc, rmse, mae, bias]

            self.identifiers[identifier]['metrics'] = metrics
    



    def fetch_datasets(self):
        pass




    def fetch_metrics(self, forecast_limit=None, long_form=False):

        self.fetch_datasets
        '''
        metrics = {}
        for i in range(len(self.identifiers)):
            forecast = self.forecasts[i]
            dataset = self.datasets[i]
            climatological = self.climatological_trimmed[i]
            metric = ValidationDataset()
            metrics[self.identifiers[i]] = metric
        self.metrics = metrics
        '''

    def threshold(self, metric, threshold, derivative_type):
        if derivative_type in ['boolean']:
            for identifier in self.identifiers:
                if metric in ['RMSE', 'MAE']:
                    if self.identifiers[identifier]['metrics'][MAP_METRICS[metric]] > threshold:
                        return True
                if metric in ['BIAS']:
                    if abs(self.identifiers[identifier]['metrics'][MAP_METRICS[metric]]) > threshold:
                        return True 
                if metric in ['ACC']:
                    if self.identifiers[identifier]['metrics'][MAP_METRICS[metric]] < threshold: 
                        return True
            return False
        elif derivative_type in ['percentage', 'number']:
            counter = 0 
            for identifier in self.identifiers:
                if metric in ['RMSE', 'MAE']:
                    if self.identifiers[identifier]['metrics'][MAP_METRICS[metric]] > threshold:
                        counter+=1
                if metric in ['BIAS']:
                    if abs(self.identifiers[identifier]['metrics'][MAP_METRICS[metric]]) > threshold:
                        counter+=1
                if metric in ['ACC']:
                    if self.identifiers[identifier]['metrics'][MAP_METRICS[metric]] < threshold: 
                        counter+=1 
            if derivative_type in ['percentage']:
                return counter/len(self.identifiers)   
            else: 
                return counter  

#query = ValidationQuery(['2022-04-20', '2022-04-21', '2022-04-22', '2022-04-23'], [[40, -120]], ['gfs_tmax-hourly'], 'rtma_temp-hourly', length_limit=5)
#print(query.threshold('RMSE', 4.5, 'percentage'))
#for identifier in query.identifiers:
#    print(query.identifiers[identifier]['metrics'])

#print(ValidationDataset(ForecastDataset('gfs_tmax-hourly', 40, -120, '2022-01-01', end_date='2022-01-03'), 'era5_land_2m_temp-hourly').get_validation(long_form=True))

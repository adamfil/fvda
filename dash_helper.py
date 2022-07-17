def readable_name(item):
    readable = {
        "chirpsc_final_05-daily": "CHIRPS Final 0.05 Resolution Precipitation",
        "chirpsc_final_25-daily": "CHIRPS Final 0.25 Resolution Precipitation",
        "chirpsc_prelim_05-daily": "CHIRPS Preliminary 0.05 Resolution Precipitation",
        "climatological_averager": "Climatological Averager",
        "cpcc_precip_global-daily": "Climate Prediction Center Global Precipitation",
        "cpcc_precip_us-daily": "Climate Prediction Center US Precipitation",
        "cpcc_temp_max-daily": "Climate Prediction Center Global Maximum Temperature",
        "cpcc_temp_min-daily": "Climate Prediction Center Global Minimum Temperature",
        "ecmwf_forecasts_temp2m-three_hourly": "European Centre for Medium-Range Weather Forecasts 2m Temperature",
        "ecmwf_forecasts_windu10m-three_hourly": "European Centre for Medium-Range Weather Forecasts East/West Wind",
        "ecmwf_forecasts_windv10m-three_hourly": "European Centre for Medium-Range Weather Forecasts North/South Wind",
        "era5_instantaneous_10m_wind_gust-hourly": "ERA5 10m Wind Gust",
        "era5_land_2m_temp-hourly": "ERA5 Land 2m Temperature",
        "era5_land_dewpoint_temp_2m-hourly": "ERA5 Land 2m Dew Point",
        "era5_land_precip-hourly": "ERA5 Land Precipitation",
        "era5_land_snowfall-hourly": "ERA5 Land Snowfall",
        "era5_land_surface_solar_radiation_downwards-hourly": "ERA5 Land Surface Solar Radiation",
        "era5_land_wind_u-hourly": "ERA5 Land East/West Wind",
        "era5_land_wind_v-hourly": "ERA5 Land North/South Wind",
        "era5_surface_runoff-hourly": "ERA5 Surface Runoff",
        "era5_volumetric_soil_water_layer_1-hourly": "ERA5 Top Layer Soil Moisture",
        "era5_wind_100m_u-hourly": "ERA5 100m East/West Wind",
        "era5_wind_100m_v-hourly": "ERA5 100m North/South Wind",
        "prismc-precip-daily": "PRISM Precipitation",
        "prismc-tmax-daily": "PRISM Maximum Temperature",
        "prismc-tmin-daily": "PRISM Minimum Temperature",
        "rtma_dew_point-hourly": "Real-Time Mesoscale Analysis Dewpoint",
        "rtma_gust-hourly": "Real-Time Mesoscale Analysis Wind Gust",
        "rtma_pcp-hourly": "Real-Time Mesoscale Analysis Precipitation",
        "rtma_temp-hourly": "Real-Time Mesoscale Analysis Temperature",
        "rtma_wind_u-hourly": "Real-Time Mesoscale Analysis East/West Wind",
        "rtma_wind_v-hourly": "Real-Time Mesoscale Analysis North/South Wind",
        "vhi": "Vegetative Health Index",
        "gfs_10m_wind_u-hourly": "Global Forecast System 10m East/West Wind",
        "gfs_10m_wind_v-hourly": "Global Forecast System 10m North/South Wind",
        "gfs_pcp_rate-hourly": "Global Forecast System Precipitation Rate",
        "gfs_relative_humidity-hourly": "Global Forecast System Relative Humidity",
        "gfs_tmax-hourly": "Global Forecast System Maximum Temperature",
        "gfs_tmin-hourly": "Global Forecast System Minimum Temperature",
        "gfs_volumetric_soil_moisture-hourly": "Global Forecast System Soil Moisture"
    }

    return readable[item]


def to_dash_list_dict(list):
    results = []
    for item in list:
        results.append({'label': readable_name(item), 'value': item})
    return results

from scipy.ndimage import label
import numpy as np
import xarray as xr
import pandas as pd
import cftime
import datetime
from tqdm import tqdm
from scipy.stats import genextreme as gev
from numpy.random import default_rng
rng = default_rng()

CESM2_REALIZATION_DICT = {
    "historical": {"A": "1500", "B": "1000", "C": "1200"},
    "SSP370": {"A": "1500", "B": "0600", "C": "0900"},
}

scen_config_dict = {
    "current": "Historical system",
    "future": "Net zero system",
    "future_wind_x2": "High wind"+", "+"Net zero system",
    "future_wind_x0.5": "High solar"+", "+"Net zero system",
    "current_electrified": "Mixed heating",
    "fully_electrified": "Electric heating",
    "historical": "Historical climate",
    "SSP370": "End-of-century climate",
}

generation_dict = {
    "PV":"PV", 
    "Wind_onshore": "Onshore wind",
    "Wind_offshore":"Offshore wind",
    "hydro_ror":"Run-of-river hydropower",
    "hydro_inflow":"Reservoir hydropower"
}
demand_dict = {
    "heating-demand":"Heating demand",
    "cooling-demand":"Cooling demand",
    "weather-insensitive_demand":"Weather-insensitive"+"\n"+"demand"
}

generation_dict_line_break = {
    "PV":"PV", 
    "Wind_onshore": "Onshore"+"\n"+"Wind",
    "Wind_offshore":"Offshore"+"\n"+"Wind",
    "hydro_ror":"Run-of-river"+"\n"+"Hydropower",
    "hydro_inflow":"Reservoir"+"\n"+"Hydropower"
}
demand_dict_line_break = {
    "heating-demand":"Heating"+"\n"+"Demand",
    "cooling-demand":"Cooling"+"\n"+"Demand",
    "weather-insensitive_demand":"Weather-insensitive"+"\n"+"Demand"
}

scen_config_dict_line_break = {
    "current": "Historical system",
    "future": "Net zero system",
    "future_wind_x2": "High wind"+"\n"+"Net zero system",
    "future_wind_x0.5": "High solar"+"\n"+"Net zero system",
    "current_electrified": "Mixed heating",
    "fully_electrified": "Electric heating",
    "historical": "Historical climate",
    "SSP370": "End-of-century climate",
}

def doy_to_noleap_datetime(year, doy, hour):
    # Create a DatetimeNoLeap object, starting from Jan 1 and adding DOY offset
    dt_noleap = cftime.DatetimeNoLeap(year, 1, 1, hour) + pd.Timedelta(days=doy - 1)
    return dt_noleap
    
def country_code_to_country_name(keys):
    country_codes = {
        "CH": "Switzerland",
        "IT": "Italy",
        "FR": "France",
        "SK": "Slovakia",
        "DE": "Germany",
        "ES": "Spain",
        "AT": "Austria",
        "SI": "Slovenia",
        "SE": "Sweden",
        "UK": "United Kingdom",
        "FI": "Finland",
        "EL": "Greece",
        "GR": "Greece", #greece has two codes for some reason
        "RO": "Romania",
        "AL": "Albania",
        "BG": "Bulgaria",
        "HR": "Croatia",
        "PT": "Portugal",
        "MK": "Macedonia",
        "RS": "Serbia",
        "CZ": "Czech Republic",
        "ME": "Montenegro",
        "BA": "Bosnia and Herzegovina",
        "HU": "Hungary",
        "IE": "Ireland",
        "PL": "Poland",
        "BE": "Belgium",
        "LV": "Latvia",
        "LT": "Lithuania",
        "XK": "Kosovo",
        "NO": "Norway",
        "DK": "Denmark",
        "EE":"Estonia",
        "NL": "Netherlands",
    }

    if type(keys) == list:
        return list( map(country_codes.get, keys) )
    if keys == None:
        return country_codes
    else:
        return country_codes[keys]

def country_name_to_country_code(keys):
    country_codes = {
        "Switzerland":"CH",
        "Italy":"IT",
        "France":"FR",
        "Slovakia":"SK",
        "Germany":"DE",
        "Spain":"ES",
        "Austria":"AT",
        "Slovenia":"SI",
        "Sweden":"SE",
        "United Kingdom":"UK",
        "Finland":"FI",
        "Greece":"EL",
        "Romania":"RO",
        "Albania":"AL",
        "Bulgaria":"BG",
        "Croatia":"HR",
        "Portugal":"PT",
        "Macedonia":"MK",
        "Serbia":"RS",
        "Czech Republic":"CZ",
        "Montenegro":"ME",
        "Bosnia and Herzegovina":"BA",
        "Hungary":"HU",
        "Ireland":"IE",
        "Poland":"PL",
        "Belgium":"BE",
        "Latvia":"LV",
        "Lithuania": "LT",
        "Kosovo": "XK",
        "Norway":"NO",
        "Denmark":"DK",
        "Estonia":"EE",
        "Netherlands":"NL",
    }
    if type(keys) == list:
        return list( map(country_codes.get, keys) )
    if keys == None:
        return country_codes
    else:
        return country_codes[keys]

def find_islands(da, threshold):
    """
    find the largest island of values in a data array da, that are above a certain threshold
    """
    da_filled = da.fillna(0)
    # Label connected components
    labeled_array, num_features = label(da_filled)
    # Calculate the size of each island
    island_sizes = np.bincount(labeled_array)
    #index and mask
    large_islands_indices = np.where(island_sizes[1:] > threshold)[0] + 1 #don't count the size of empty
    large_islands_masks = [labeled_array == isl_ind for isl_ind in large_islands_indices]
    #return all large islands
    large_islands = [da.where(mask).dropna("time") for mask in large_islands_masks]
    return large_islands

def find_longest_islands(arr):
    # Find change points (start and end of 1s)
    nonzero_mask = arr != 0
    change_points = np.diff(np.concatenate(([0], nonzero_mask, [0])))
    start_indices = np.where(change_points == 1)[0]
    end_indices = np.where(change_points == -1)[0] - 1
    # Compute durations
    durations = np.zeros_like(arr)
    durations[end_indices] = end_indices - start_indices + 1
    # Create output array with only last value of each sequence
    output = np.zeros_like(arr)
    output[end_indices] = arr[end_indices]
    return output,durations

def fit_gev(data):
    """
    Fit a GEV law to a data set 
    """
    # Fit the GEV distribution to your data and return the shape, location, and scale
    shape, loc, scale = gev.fit(data)
    return shape, loc, scale

def find_return_time_naive_gev(dataset,return_level,bootstrap=1000):
    """
    find the return time of a return level with a GEV law fitted to a dataset, with uncertainty given by bootstrapping
    """
    bootstrapped_dataset= xr.DataArray(data = rng.choice(dataset, size=(bootstrap, len(dataset)), replace=True))
    bootstrap_return_time = []
    for i in tqdm(range(bootstrap)):
        shape, loc, scale = fit_gev(bootstrapped_dataset.sel(dim_0=i))
        prob = (1-gev.cdf(return_level, shape, loc=loc, scale=scale))
        if prob == 0:
            ret = float('inf')
        else:
            ret = 1/prob
        bootstrap_return_time.append(ret)
    print(f"{np.median(bootstrap_return_time):.5},{np.quantile(bootstrap_return_time,0.025):.5},{np.quantile(bootstrap_return_time,0.975):.5}")
    return bootstrap_return_time

def get_time_range(scenario):
    """
    Start and end years of the different scenarios covered in this analysis
    :param scenario:
    :return:
    """
    range_dict = {
        "historical": range(1995, 2015),
        "SSP370": range(2080, 2100),
        "SSP245": range(2015, 2100),
    }
    return range_dict[scenario]

def select_Europe(ds):
    return ds.sel(lon=slice(-15, 50), lat=slice(30, 75))

def zero_mean_longitudes(ds):
    """
    resort a dataset with longitudes from
        0 to 360
    to one that has longitudes from
        -180 to 180
    :param ds:
    :return:
    """
    ds.coords["lon"] = (ds.coords["lon"] + 180) % 360 - 180
    ds = ds.sortby("lon")
    return ds

def get_time_range_noleap(start_year,end_year):    
    # Total hours = 20 years × 365 days/year × 24 hours/day
    n_hours = ( end_year - start_year ) * 365 * 24
    
    # Generate hourly time range
    start = cftime.DatetimeNoLeap(start_year, 1, 1, 0)
    return np.array([start + datetime.timedelta(hours=i) for i in range(n_hours)])

def get_smoothed_doy(doy,roll):
    doy_year_before = doy.copy()
    doy_year_before["dayofyear"] = doy["dayofyear"].values - 365
    doy_year_after = doy.copy()
    doy_year_after["dayofyear"] = doy["dayofyear"].values + 365
    return xr.concat([doy_year_before,doy, doy_year_after],dim="dayofyear").rolling(dayofyear=roll,center=True).mean().sel(dayofyear=slice(1,365))

def get_minmax(data):
    min_local = data.min()
    max_local=data.max()
    max_local= np.max([np.abs(min_local),np.abs(max_local)])
    min_local = -max_local
    return min_local,max_local
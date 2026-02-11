from scipy.ndimage import label
import numpy as np
import xarray as xr
import pandas as pd
import cftime
import datetime
import geopandas as gpd
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

tech_dict = {
    "PV":"PV", 
    "Wind_onshore": "Onshore wind",
    "Wind_offshore":"Offshore wind",
    "hydro_ror":"Run-of-river hydropower",
    "hydro_inflow":"Reservoir hydropower",
    "heating-demand":"Heating demand",
    "cooling-demand":"Cooling demand",
    "weather-insensitive_demand":"Weather-insensitive"+"\n"+"demand"
}

generation_dict_line_break = {
    "PV":"PV", 
    "Wind_onshore": "Onshore"+"\n"+"Wind",
    "Wind_offshore":"Offshore"+"\n"+"Wind",
    "hydro_ror":"Run-of-river"+"\n"+"Hydropower",
    "hydro_inflow":"Reservoir"+"\n"+"Hydropower",
    "hydro_dispatched":"Reservoir"+"\n"+"Hydropower",
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
    "historical": "Historical"+"\n"+"climate",
    "SSP370": "End-of-century"+"\n"+"climate",
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

shapefile_path = "/home/lbloin/Thesis/eng_boost/net_load_boosting/inputs/geopandas/ne_50m_admin_0_countries.shp"
# Load the shapefile
world = gpd.read_file(shapefile_path)
# Filter for Europe
europe = world[world['CONTINENT'] == 'Europe']

def country_to_region(country,europe):
    subregion = europe[europe["NAME_LONG"]==country]["SUBREGION"].item()
    return subregion

def get_region_mean(ds):
    region = xr.DataArray(
        [country_to_region(c,europe) for c in ds.country.values],
        dims="country",
        coords={"country": ds.country},
        name="region"
    )
    return ds.groupby(region).mean("country")

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
    return ds.sel(lon=slice(-25, 35), lat=slice(30, 75))

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

def get_time_plus_delta(start_date,delta,delta_minus=0):    
    # Total hours 
    n_hours = delta * 24
    nhours_minus = delta_minus *24
    # Generate hourly time range
    start_year = int(start_date[0:4])
    start_month = int(start_date[5:7])
    start_day = int(start_date[8:10])
    start = cftime.DatetimeNoLeap(start_year, start_month, start_day, 0)
    return slice(start - datetime.timedelta(hours=nhours_minus), start + datetime.timedelta(hours=n_hours))

def get_smoothed_doy(doy,roll):
    doy_year_before = doy.copy()
    doy_year_before["dayofyear"] = doy["dayofyear"].values - 365
    doy_year_after = doy.copy()
    doy_year_after["dayofyear"] = doy["dayofyear"].values + 365
    return xr.concat([doy_year_before,doy, doy_year_after],dim="dayofyear").rolling(dayofyear=roll,center=True).mean().sel(dayofyear=slice(1,365))

def get_smoothed_hoy(hoy,roll):
    hoy_year_before = hoy.copy()
    hoy_year_before["hourofyear"] = hoy["hourofyear"].values - 8760
    hoy_year_after = hoy.copy()
    hoy_year_after["hourofyear"] = hoy["hourofyear"].values + 8760
    return xr.concat([hoy_year_before,hoy, hoy_year_after],dim="hourofyear").rolling(hourofyear=roll,center=True).mean().sel(hourofyear=slice(0,8759))

def get_minmax(data):
    min_local = data.min()
    max_local=data.max()
    max_local= np.max([np.abs(min_local),np.abs(max_local)])
    min_local = -max_local
    return min_local,max_local

def multi_to_single_index(ds,dims=("case","lead_time"),new_name="lead_ID"):
    stacked = ds.stack(event=dims)
    coords = [str(st) for st in stacked["event"].values]
    stacked = stacked.reset_index("event")
    stacked["event"] = coords
    stacked = stacked.drop_vars(dims)
    stacked = stacked.rename({"event":new_name})
    return stacked

def ds_hoy_in_full_time(ds,typ,dims=("member","time")):
    """get hour of year values but with time coordinates too"""
    hour_of_year = xr.DataArray(
        np.arange(len(ds.time)) % 8760,
        dims="time",
        coords={"time": ds.time}
    )
    # Add it as a coordinate for easy grouping
    ds = ds.assign_coords(hourofyear=hour_of_year)
    # mean or std
    if typ =="mn":
        flattened = ds.groupby("hourofyear").mean(dims)
    elif typ =="std":
        flattened = ds.groupby("hourofyear").std(dims)
    flattened = get_smoothed_hoy(flattened,24*7)
    time = ds.time
    hoy = time.hourofyear
    return flattened.sel(hourofyear=hoy).assign_coords(time=time)

def str_to_cftime_noleap(string):
    return cftime.DatetimeNoLeap(int(string[0:4]), int(string[5:7]), int(string[8:10]))
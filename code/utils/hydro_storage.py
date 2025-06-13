import utils as ut
import xarray as xr
from tqdm import tqdm
import numpy as np
import pandas as pd
import cftime
import datetime
import glob


def open_storage(scenario):
    """
    Opens the storage data set from Francesco's energy model for a given scenario, into the format used in the rest of the project
    Here, all three parallel members (A,B,C) are stacked on top of each other, and each year (60 in total, 3*20 parallel simulations) is called a scenario.
    This is not to be confused with our parameter scenario, which refers to either historical or SSP370.
    :param scenario: either historical or SSP245
    """
    file = glob.glob(f"/net/xenon/climphys/lbloin/energy_boost/storage_level_reservoir_hydro_{scenario}.nc")
    try :
        xr.open_dataset(file[0])["storage"]
    except:
        # open pandas df 
        df = pd.read_csv(f"../inputs/storage_level_reservoir_hydro_{scenario}.csv", index_col=[0,1,2]).droplevel('technology')
        # NB: scenario here is the parallel simulations. this will change below
        df.index.names = ['scenario', 'country']
        df.index = df.index.set_levels(
            [
                df.index.levels[0].str.extract(r'(\d+)')[0].astype(int)
                if level.name == 'scenario' else level
                for level in df.index.levels
            ]
        )
        # sort the index
        df = df.sort_index(level='scenario')

        # Parameters
        members = ['A', 'B', 'C']
        years_per_member = 20
        total_sims = years_per_member * len(members)  # 60 parallel simulation total
        time_per_year = 8760  # hours per year (no leap years)
        if scenario == "historical":
            start_year = 1995
        else:
            start_year = 2080
        
        # Extract the simulation indices and nodes from DataFrame index
        sim_indices = df.index.levels[0].astype(int)
        nodes = df.index.levels[1]
        
        # Create arrays for member and year corresponding to each parallel simulation
        members_arr = np.repeat(members, years_per_member)   # ['A']*20 + ['B']*20 + ['C']*20
        years_arr = np.tile(np.arange(start_year, start_year + years_per_member), len(members))  # repeated for each member
        
        # Create xarray DataArray from the DataFrame values
        # Reshape df.values from (parallel_sims, nodes, time) => (60, n_nodes, 8760)
        da = xr.DataArray(
            df.values.reshape(total_sims, len(nodes), time_per_year),
            dims=['parallel_sims', 'country', 'time'],
            coords={
                'parallel_sims': sim_indices,
                'country': ut.country_code_to_country_name(list(nodes)),
                'time': df.columns.astype(int)
            }
        )
        # assign member and year coordinates to the 'parallel_sims' dimension
        da = da.assign_coords(
            member=('parallel_sims', members_arr),
            year=('parallel_sims', years_arr)
        )
        # unstack the 'parallel_sims' dimension into separate 'member' and 'year' dimensions
        da = da.set_index(parallel_sims=['member', 'year']).unstack('parallel_sims')
        da_stacked = da.stack(year_time=("year", "time")).drop_vars(['year_time', 'time', 'year'])
        # For 20 years of hourly data, no leap days
        times = []
        for year in range(start_year, start_year + years_per_member):
            for hour in range(time_per_year):
                delta = datetime.timedelta(hours=hour)
                times.append(cftime.DatetimeNoLeap(year, 1, 1) + delta)
        # Assign the combined datetime coordinate
        da_stacked = da_stacked.assign_coords(year_time=times)
        # # Rename the stacked dimension to 'time' for clarity
        ds_storage = da_stacked.rename({'year_time': 'time'}).to_dataset(name="storage")
        ds_storage.to_netcdf(f"/net/xenon/climphys/lbloin/energy_boost/storage_level_reservoir_hydro_{scenario}.nc")
        return ds_storage["storage"]


def calculate_storage_net_load(inflow_t, net_load_t, storage_t_minus_1, net_load_thresh, mean,std,storage_max,capacity_thresh):
    """
    Calculates the storage level in hydro reservoirs, and returns the storage level and adjusted net load at time t 
    Args: 
        inflow_t: inflow at time t (float)
        net_load_t: net load at time t, without storage adjustment (float)
        storage_t_minus_1: storage level at time t-1 (float)
        net_load_thresh: threshold at which net_load becomes critical enough that storage is needed (float)
        storage_thresh: values at time t of different storage levels, to create a mean and standard deviation (np.arr())
    """
    #fill storage unless its full
    if storage_t_minus_1 >= storage_max:
        storage_t = storage_t_minus_1
        net_load_t = net_load_t - inflow_t
    else:
        storage_t = storage_t_minus_1 + inflow_t #fill storage with inflow at time t
    if net_load_t-net_load_thresh > 0: # if net load is above threshold, we need to adjust the storage and net load levels
        #calculate mean and std dev storage thresholds for time t
        upper_storage_thresh = mean + std
        lower_storage_thresh = mean - std
        # case 1: storage is fuller than average for that time of year 
        if  storage_t - upper_storage_thresh > 0 :
            # we remove either the necessary net load to go under the threshold, 
            # or if there isn't enough storage, the remaining storage until the upper storage threshold is reached
            take_from_storage = min((net_load_t-net_load_thresh),capacity_thresh,storage_t-upper_storage_thresh)
            storage_t = storage_t - take_from_storage
            net_load_t = net_load_t - take_from_storage
        # case 2: storage is typical for that time of year
        if storage_t - upper_storage_thresh <= 0 and storage_t - lower_storage_thresh > 0:
            # we remove either the half of the necessary net load to go under the threshold, 
            # or if there isn't enough storage, half of the remaining storage between the upper and lower threshold
            take_from_storage = min((net_load_t-net_load_thresh), capacity_thresh/2, storage_t-lower_storage_thresh) #(upper_storage_thresh - lower_storage_thresh)/(7*24)
            storage_t = storage_t - take_from_storage
            net_load_t = net_load_t - take_from_storage 

        #  case 3: storage is emptier than average for that time of year
        if storage_t - lower_storage_thresh <= 0 and storage_t >0:
            # we remove either the a quarter of the necessary net load to go under the threshold, 
            # or if there isn't enough storage, a quarter of the remaining storage between the upper and lower threshold
            take_from_storage = min((net_load_t-net_load_thresh), capacity_thresh/10,storage_t)
            storage_t = storage_t - take_from_storage
            net_load_t = net_load_t - take_from_storage 

    storage_out = storage_t
    net_load_out = net_load_t
    return storage_out,net_load_out


def calculate_storage_net_load_country(net_load, hydro_inflow,storage_roll,storage_max,starting_storage,qu,max_capac):
    """
    Calculates the storage effects for all countries.
    """
    storage = []
    net_load_adj = []
    for country in net_load.country:
        # select country-specific net load
        net_load_country = net_load.sel(country=country).values
        if country.values in hydro_inflow.country.values:
            #select other country-specific variables
            hif_country = hydro_inflow.sel(country=country)
            hif_country = hif_country.where(hif_country >0, 0).values
            net_load_thresh_country = qu.sel(country=country).item()
            capacity_thresh_country = max_capac.sel(country=country).item()
            storage_thresh_country = storage_roll.sel(country=country).groupby("time.dayofyear")
            storage_thresh_mean = storage_thresh_country.mean().values
            storage_thresh_std = storage_thresh_country.std().values
            storage_max_country = storage_max.sel(country=country).item()
            # loop over time
            storage_country = [starting_storage.sel(country=country).item()] #storage at time 0 stays the same
            net_load_adj_country = [net_load_country[0]] #adjusted net load at time 0 stays the same
            for i in (range(1, len(net_load_country))):
                doy = int(i%8760/24) # day of year value is hour of year value /24
                if i ==0:
                    continue
                else:
                    #calculate adjusted net load and storage values
                    storage_t, net_load_adj_t = calculate_storage_net_load(
                        hif_country[i],  
                        net_load_country[i], 
                        storage_country[i-1], 
                        net_load_thresh_country, 
                        storage_thresh_mean[doy],
                        storage_thresh_std[doy],
                        storage_max_country,
                        capacity_thresh_country
                    )
                    storage_country.append(storage_t)
                    net_load_adj_country.append(net_load_adj_t)
            storage.append(storage_country)
            net_load_adj.append(net_load_adj_country)
        else:
            storage.append(list(np.zeros(len(net_load_country)))) # no storage if no hydro
            net_load_adj.append(net_load_country)
    net_load_adj = xr.DataArray(net_load_adj,dims=["country","time"], coords={"time":net_load.time,"country":net_load.country})
    storage = xr.DataArray(storage,dims=["country","time"], coords={"time":net_load.time,"country":net_load.country})
    out_ds = storage.to_dataset(name="storage")
    out_ds["net_load_adjusted"] = net_load_adj
    return out_ds

def storage_net_load_all_dims(abs_vars_tech_sum,techs,hydro_inflow_full,storage_roll,storage_max,starting_storage):
    """
    Calculates storage effects over all dimensions of the net load dataset (technology capacity scenario, heating scenario and member)"""
    adjusted_net_load_capac = []
    for capacity_scenario in tqdm(abs_vars_tech_sum.capacity_scenario.values):
        adjusted_net_load_heat_scenario = []
        for heat_scenario in techs["heating-demand"]:
            adjusted_net_load_mem = []
            # 75th quantile, used as the threshold for when hydro storage is needed
            qu = abs_vars_tech_sum.sel(capacity_scenario=capacity_scenario,heating_scenario=heat_scenario).quantile(0.75,dim=("member","time"))
            for member in abs_vars_tech_sum.member.values:
                #select specific realization of net load and inflow datasets
                net_load_specific = abs_vars_tech_sum.sel(capacity_scenario=capacity_scenario,member=member,heating_scenario=heat_scenario)
                hydro_inflow = hydro_inflow_full.sel(capacity_scenario=capacity_scenario,member=member,heating_scenario=heat_scenario)
                # calculate adjusted net load and storage
                adjusted_net_load=calculate_storage_net_load_country(net_load_specific, hydro_inflow,storage_roll,storage_max,starting_storage,qu)
                adjusted_net_load_mem.append(adjusted_net_load)
            adjusted_net_load_heat_scenario.append(xr.concat(adjusted_net_load_mem,dim=pd.Index(abs_vars_tech_sum.member.values, name="member")))
        adjusted_net_load_capac.append(xr.concat(adjusted_net_load_heat_scenario,dim=pd.Index(list(techs["heating-demand"].keys()), name="heating_scenario")))
    return xr.concat(adjusted_net_load_capac,dim=pd.Index(abs_vars_tech_sum.capacity_scenario.values, name="capacity_scenario"))




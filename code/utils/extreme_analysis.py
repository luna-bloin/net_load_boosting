import xarray as xr
import pandas as pd
import numpy as np
import math
import glob
#local imports
import utils as ut
import plot_config as pco
#plotting
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import geopandas as gpd
#others
from datetime import datetime, timedelta
import cftime


# =====================
# === Opening files ===
# =====================

def open_joint_clim(name,var,qu=False,country_sum=False):
    """opens a type of net load (given by name and var), and returns it, as well as its 90th percentile (optional). If country_sum is true, it sums over all countries"""
    ds_hist = xr.open_dataset(f"{name}_historical.nc").rolling(time=24,center=True).mean()
    ds_eoc = xr.open_dataset(f"{name}_SSP370.nc").rolling(time=24,center=True).mean()
    if country_sum == True:
        ds_hist = ds_hist.sum("country")
        ds_eoc =ds_eoc.sum("country")
    ds = xr.concat([ds_hist, ds_eoc],dim=pd.Index(["historical","SSP370"], name="climate"),join="outer")
    if qu == True:
        ds_qu = ds.quantile(0.9,("time","member","climate"))
        return ds[var], ds_qu[var]
    else:
        return ds[var]

def open_all_parent_nl(path,nl_only=False):
    # Open net load with transmission
    nl, nl_qu = open_joint_clim(f"{path}net_load_transmission","net_load",qu=True)
    capacity_scenarios = nl.capacity_scenario.values
    heating_scenarios = nl.heating_scenario.values
    # find extremes
    extremes = find_extremes_all_scenarios(nl,nl_qu,heating_scenarios,capacity_scenarios)
    if nl_only ==True:
        return nl,nl_qu,extremes
    # open tech by tech data
    file = glob.glob(f"{path}tech_nl.nc")
    if len(file)>0:
        tech_nl = xr.open_dataset(file[0])["tech_nl"]
    else:
        tech_nl = open_joint_clim(f"{path}eng_vars_GWh","eng_vars",country_sum=True)
        mns_tech = ut.ds_hoy_in_full_time(tech_nl,"mn",dims=("time","member","climate"))
        stds_tech = ut.ds_hoy_in_full_time(tech_nl,"std",dims=("time","member","climate"))
        tech_nl =  xr.concat([tech_nl,mns_tech,stds_tech],dim=pd.Index(["full","mean","std"], name="value_type"))
        tech_nl.to_dataset(name="tech_nl").to_netcdf(f"{path}tech_nl.nc")
    # open storage data
    file = glob.glob(f"{path}storage_nl.nc")
    if len(file)>0:
        storage_nl = xr.open_dataset(file[0])["storage_nl"]
    else:
        storage_nl = open_joint_clim(f"{path}net_load_hydro_storage","storage")
        mns_stor = ut.ds_hoy_in_full_time(storage_nl,"mn",dims=("time","member","climate"))
        stds_stor = ut.ds_hoy_in_full_time(storage_nl,"std",dims=("time","member","climate"))
        storage_nl =  xr.concat([storage_nl,mns_stor,stds_stor],dim=pd.Index(["full","mean","std"], name="value_type"))
        storage_nl.to_dataset(name="storage_nl").to_netcdf(f"{path}storage_nl.nc")
    # open by country data
    file = glob.glob(f"{path}region_nl.nc")
    if len(file)>0:
        region_nl = xr.open_dataset(file[0])["region_nl"]
    else:
        country_nl = open_joint_clim(f"{path}net_load_by_country_hydro_storage","net_load_adjusted")
        region_nl = ut.get_region_mean(country_nl.drop_sel(country="Macedonia"))
        mns_region = ut.ds_hoy_in_full_time(region_nl,"mn",dims=("time","member","climate"))
        stds_region = ut.ds_hoy_in_full_time(region_nl,"std",dims=("time","member","climate"))
        region_nl =  xr.concat([region_nl,mns_region,stds_region],dim=pd.Index(["full","mean","std"], name="value_type"))
        region_nl.to_dataset(name="region_nl").to_netcdf(f"{path}region_nl.nc")
    return nl, nl_qu,extremes, tech_nl, region_nl, storage_nl

def open_atm_vars(path):
    dss_scen = {}
    dss_mn = []
    for scenario in ut.CESM2_REALIZATION_DICT:
        years = ut.get_time_range(scenario)
        dss = []
        for var in ["temperature","s_hub","Z500","global_horizontal"]:
            dss.append(xr.open_dataset(f"{path}bced_{var}_{scenario}.nc").sel(time=slice(str(years.start), str(years.stop-1))))
        dss = xr.merge(dss)
        dss_scen[scenario] = dss
        dss_mn.append(dss.groupby("time.dayofyear").mean(("time","member")))
    dss_mn = xr.concat(dss_mn,dim="climate").mean("climate")
    return dss_scen,dss_mn

# =============================
# === Opening boosted files ===
# =============================

def open_boost(path,boost_dates,start_parent,scenario,member,typ="transmission"):
    """opens different types of net load for boosted events (given a set of boosted dates of perturbation) and concatenates it to its parent for continuity. the parent starts at start_parent (a date that makes sure to include all of the parent extreme)."""
    boost = []
    for date in boost_dates:
        if typ == "atm":
            ds = []
            for var in ["temperature","s100","Z500","global-horizontal"]:
                ds.append(xr.open_dataset(f"{path}bced_{var}_{scenario}_boost_{member}_{date}.nc"))
            ds = xr.merge(ds)
            boost.append(ds)
        else:
            if typ =="transmission":
                name = "net_load_transmission"
                parent = xr.open_dataset(f"{path}{name}_{scenario}.nc").net_load
                ds = xr.open_dataset(f"{path}{name}_{scenario}_boost_{member}_{date}.nc").net_load
            elif typ =="simple":
                name = "net_load_simple"
                parent = xr.open_dataset(f"{path}{name}_{scenario}.nc").net_load
                ds = xr.open_dataset(f"{path}{name}_{scenario}_boost_{member}_{date}.nc").net_load
            elif typ == "eng_vars":
                name = "eng_vars_GWh_country_sum"
                parent = xr.open_dataset(f"{path}{name}_{scenario}.nc").eng_vars
                ds = xr.open_dataset(f"{path}{name}_{scenario}_boost_{member}_{date}.nc").eng_vars
            elif typ == "region":
                name = "net_load_by_country_hydro_storage"
                parent = ut.get_region_mean(xr.open_dataset(f"{path}{name}_{scenario}.nc").net_load_adjusted.drop_sel(country="Macedonia"))
                ds = ut.get_region_mean(xr.open_dataset(f"{path}{name}_{scenario}_boost_{member}_{date}.nc").net_load_adjusted.drop_sel(country="Macedonia"))
            ds = ds.sel(time=slice(date,None))
            parent_here = parent.sel(member=member).drop_vars("member").broadcast_like(ds)
            parent_here = parent_here.sel(time=slice(start_parent, ut.str_to_cftime_noleap(date)-timedelta(hours=1))) #prepare parent for concat
            boost_here = xr.concat([parent_here,ds],dim="time")
            boost_here = boost_here.rolling(time=24,center=True).mean()
            boost.append(boost_here)
    boost = xr.concat(boost,dim="lead_time",join="outer")
    boost["lead_time"] = boost_dates
    return boost

def find_start_end_boost(nl_boost_here,event,start,qu_parent):
    to_plot_highlight = nl_boost_here.stack(event=("lead_time","member")).sel(event=event).dropna(dim="time",how="all").sel(time=slice(start,None))/1000 #select only time period from when parent event starts
    to_plot_highlight = to_plot_highlight.where(to_plot_highlight > qu_parent/1000) # find time steps above threshold
    if len(to_plot_highlight.dropna(dim="time",how="all")) >0: # only if event exists
        to_plot_highlight = to_plot_highlight.sel(time=slice(to_plot_highlight.dropna(dim="time",how="all").time.values[0],None)) # start where boosted energy shortfall event starts
        to_plot_highlight = to_plot_highlight.isel(time=slice(0,to_plot_highlight.isnull().argmax("time").item())) # end where energy shortfall event ends
    return to_plot_highlight

def find_dur_cum_boost(boost_to_plot,qu_parent):
    ds_lead = []
    for lead in boost_to_plot.lead_time:
        ds_member = []
        for mem in boost_to_plot.member:
            cum_boost,dur_boost = spa_algo(boost_to_plot.sel(lead_time=lead,member=mem).dropna(dim="time"),qu_parent,dim="time")
            cum_boost_max= cum_boost.max()/1000 # in TWh
            dur_boost_max = dur_boost.sel(time=cum_boost.idxmax())/24 # number of days, not hours
            ds_member.append(xr.concat([dur_boost_max,cum_boost_max],dim=pd.Index(["dur","cum"], name="typ"),join='outer'))
        ds_lead.append(xr.concat(ds_member,dim=pd.Index(boost_to_plot.member.values, name="member"),join='outer'))
    return xr.concat(ds_lead,dim=pd.Index(boost_to_plot.lead_time.values, name="lead_time"),join='outer')

# =====================
# === SPA Algorithm ===
# =====================

def above_thresh(ds,qu,dim="time"):
    """returns cumulative sums of periods above threshold (qu) for a given xarray dataset ds"""
    data_nl = (ds-qu).where(ds - qu > 0,0)
    return data_nl.cumsum(dim=dim)-data_nl.cumsum(dim=dim).where(data_nl.values == 0).ffill(dim=dim).fillna(0)

def spa_algo(ds, qu,dim="time"):
    """Calculates all periods of cumulative exceedance of a threshold (qu), and returns the final value of each period, as well as its duration (in time steps)"""
    data_nl = above_thresh(ds,qu,dim=dim)
    cumulative_nl,duration = xr.apply_ufunc(
        ut.find_longest_islands,
        data_nl,
        input_core_dims =[[dim]],
        output_core_dims=[[dim],[dim]],
    )
    return cumulative_nl, duration

def find_all_extremes_one_scenario(nl_one_scen, qu):
    """"find all extremes (events that last longer than 1 day) for a given net load scenario and quantile value"""
    nl_stacked = nl_one_scen.stack(event=("member","time"))
    cumulative_nl,duration = spa_algo(nl_stacked, qu,dim="event")
    cum = cumulative_nl.sortby(cumulative_nl,ascending=False)
    dur = duration.where(duration>24,drop=True)/24 # number of days, not hours
    cum = cum.where(duration>24,drop=True)/1000 # in TWh
    return dur, cum

def find_extremes_all_scenarios(nl,nl_qu,heating_scenarios,capacity_scenarios):
    """find all extremes in all scenarios for a given quantile threshold"""
    file = "/net/xenon/climphys/lbloin/energy_boost/all_extremes.nc"
    extremes = glob.glob(file)
    if extremes == []:
        for i,heat in enumerate(heating_scenarios):
            extreme_capac = []
            for j,capac in enumerate(capacity_scenarios):
                print(heat, capac)
                extreme_clim = []
                for x,climate in enumerate(['historical','SSP370']):
                    nl_one_scen = nl.sel(heating_scenario=heat,capacity_scenario=capac,climate=climate).drop_vars(("capacity_scenario","heating_scenario", "climate"))
                    qu = nl_qu.sel(capacity_scenario=capac,heating_scenario=heat).drop_vars(("capacity_scenario","heating_scenario","quantile"))
                    dur, cum = find_all_extremes_one_scenario(nl_one_scen, qu)
                    extreme_clim.append(xr.concat([dur,cum],dim=pd.Index(["dur","cum"], name="typ"),join='outer'))
                extreme_capac.append(xr.concat(extreme_clim,dim=pd.Index(["historical","SSP370"], name="climate"),join='outer'))
            extremes.append(xr.concat(extreme_capac,dim=pd.Index(capacity_scenarios, name="capacity_scenario"),join='outer'))
        extremes = xr.concat(extremes,dim=pd.Index(heating_scenarios, name="heating_scenario"),join='outer')
        extremes.unstack().to_dataset(name="extreme_val").to_netcdf(file)
        return extremes
    else:
        return xr.open_dataset(extremes[0])["extreme_val"].stack(event=("member","time"))

def find_nl_top(ds, top_cumul,dur,boost=False,boost_end=False):
    end = top_cumul.time.item()
    mem = top_cumul.member.item()
    duration = dur.sel(time=end,member=mem).item()
    start = end - timedelta(days=int(duration)+1)
    if boost == False:
        if boost_end != False:
            end = boost_end
        return ds.sel(time=slice(start,end),member=mem)
    else:
        return ds.sel(time=slice(start,None))

def find_peak(nl_here,mem,end,duration):
    start = end - timedelta(days=int(duration))
    peak = nl_here.sel(member=mem,time=slice(start,end)).max().item()
    return peak

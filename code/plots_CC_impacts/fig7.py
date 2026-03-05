import sys
sys.path.append("utils/")
import utils as ut
import xarray as xr
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import hydro_storage as hs
import cftime
import datetime
import glob
import plot_config as pco
import string

# =================
# === Functions ===
# =================
def plot_hydro_storage(countries=["France","Norway"],fsize=(20,5),heating_scenario = "fully_electrified",capacity_scenario = "future",scenario = "historical"):
    f1,ax1=plt.subplots(1,int(len(countries)),figsize=fsize,sharex=True)
    f2,ax2=plt.subplots(1,int(len(countries)),figsize=fsize,sharex=True)
    b_adjust = 0.25
    ax_iter = [0]
    for i,country in enumerate(countries):
        net_ax = ax1[i%5]
        stor_ax = ax2[i%5]
        nl_hydro_here = nl[scenario]["hydro_by_country"].sel(heating_scenario=heating_scenario,capacity_scenario=capacity_scenario,country=country)
        # plot net load (with and without hydro storage)
        nl_hydro_nl = nl_hydro_here.net_load_adjusted.groupby("time.dayofyear")
        mn = nl_hydro_nl.mean(("time","member"))
        std = nl_hydro_nl.std(("time","member"))
        mn.plot(ax=net_ax,zorder=3,color=pco.colors[0],label="Net load with hydro storage")
        net_ax.fill_between(mn.dayofyear.values,mn+std,mn-std,color=pco.colors[0],alpha=0.3)
        
        nl_simple = nl[scenario]["simple_by_country"].sel(heating_scenario=heating_scenario,capacity_scenario=capacity_scenario,country=country).groupby("time.dayofyear")
        mn = nl_simple.mean(("time","member"))
        std = nl_simple.std(("time","member"))
        mn.plot(ax=net_ax,color=pco.colors[3],label="Net load without hydro storage")
        net_ax.fill_between(mn.dayofyear.values,mn+std,mn-std,color=pco.colors[3],alpha=0.3)
        
        # plot average storage and storage used here
        mn = storage_doys[scenario].sel(mean_std="mean",country=country).groupby("time.dayofyear").mean()
        std = storage_doys[scenario].sel(mean_std="std",country=country).groupby("time.dayofyear").mean()
        mn.plot(ax=stor_ax,color="k",label="Average storage level")
        stor_ax.fill_between(mn.dayofyear.values,mn-std,mn+std,color="k",alpha=0.3)
    
        nl_hydro_storage = nl_hydro_here.storage.groupby("time.dayofyear")
        mn = nl_hydro_storage.mean(("time","member"))
        std = nl_hydro_storage.std(("time","member"))
        mn.plot(ax=stor_ax,color=pco.colors[6],label="Storage level with dispatch model")
        stor_ax.fill_between(mn.dayofyear.values,mn+std,mn-std,color=pco.colors[6],alpha=0.3)

        net_ax.set_title(country)
        stor_ax.set_title("")
        net_ax.set_xlabel("")
        stor_ax.set_xlabel("")
        net_ax.set_xticks([60,152,244,335],labels=["Mar","Jun","Sep","Dec"])
        stor_ax.set_xticks([60,152,244,335],labels=["Mar","Jun","Sep","Dec"])
    
    f = [f1,f2]
    ylab= ["Net load [GW]","Storage level [GWh]"]
    for j,ax in enumerate([ax1,ax2]):
        for i,a in enumerate(ax.flatten()):
            pco.set_grid(a)
            a.text(0.01,0.9,string.ascii_lowercase[i+j*2],weight="bold",transform=a.transAxes)
            a.set_ylabel("")
        ax[0].set_ylabel(ylab[j])
        handles, labels = ax[0].get_legend_handles_labels()
        f[j].legend(handles, labels, loc='lower center',ncol=2)
        f[j].tight_layout()
        f[j].subplots_adjust(bottom=b_adjust)
    return f1,ax1,f2,ax2

# ==================
# === Open files ===
# ==================

path = "/net/xenon/climphys/lbloin/energy_boost/"
# get net load for all scenarios/climate periods, with different resolution (by technology, all together, with transmission assumption)
nl = {}
for scenario in  ut.CESM2_REALIZATION_DICT:
    net_load_simple = xr.open_dataset(f"{path}net_load_simple_{scenario}.nc").net_load  
    nl_hydro = xr.open_dataset(f"{path}net_load_hydro_storage_{scenario}.nc")
    net_load_hydro = nl_hydro.net_load_adjusted
    storage = nl_hydro.storage
    net_load_simple_by_tech= xr.open_dataset(f"{path}eng_vars_GWh_country_sum_{scenario}.nc").eng_vars
    net_load_simple_by_country = xr.open_dataset(f"{path}net_load_by_country_simple_{scenario}.nc").net_load 
    net_load_hydro_by_country = xr.open_dataset(f"{path}net_load_by_country_hydro_storage_{scenario}.nc") 
    nl_scen = {"simple":net_load_simple,"simple_by_country":net_load_simple_by_country,"hydro": net_load_hydro,"hydro_by_country": net_load_hydro_by_country,"by_tech":net_load_simple_by_tech,"storage":storage}
    nl[scenario] = nl_scen

# get storage as hour of year per country
storage_doys = {}
for scenario in  ut.CESM2_REALIZATION_DICT:
    storage = hs.open_storage(scenario,'')
    # Convert to DataArray of hour-of-year values (0–8759)
    hour_of_year = xr.DataArray(
        np.arange(len(storage.time)) % 8760,
        dims="time",
        coords={"time": storage.time}
    )
    # Add it as a coordinate for easy grouping
    storage = storage.assign_coords(hourofyear=hour_of_year)
    storage_roll= storage.rolling(time=24*21,center=True).mean()# rolling average to smooth the curve
    # calculate mean and std dev
    mean_vals = storage_roll.groupby("hourofyear").mean(("time","member")).values
    std_vals = storage_roll.groupby("hourofyear").std(("time","member")).values
    storage_doy = []
    for vals in [mean_vals,std_vals]:
        da = xr.DataArray(
            data=np.tile(vals,20),
            dims=['country', 'time'],
            coords={
                'country': storage.country,
                'time': net_load_hydro["time"].values[0:-1]
            },
            name='mean_storage'
        )
        storage_doy.append(da)
    storage_doys[scenario] = xr.concat(storage_doy,dim=pd.Index(["mean","std"], name="mean_std"))

heating_scenario ="fully_electrified"
capacity_scenario = "future"
scenario = "historical"

# ===============================
# === Appendix figure Panel e ===
# ===============================

f,axs=plt.subplots(figsize=(7,3))
nl_inst_here = (nl[scenario]["by_tech"].sel(heating_scenario=heating_scenario,capacity_scenario=capacity_scenario,technology="hydro_inflow").groupby("time.dayofyear"))
mn = nl_inst_here.mean(("member","time"))
std =nl_inst_here.std(("member","time"))
mn.plot(color=pco.colors[3],linestyle="solid",label="Instantaneous reservoir hydropower")
axs.fill_between(mn.dayofyear.values,mn-std,mn+std,color=pco.colors[3],alpha=0.3)

nl_simple = nl[scenario]["simple"].sel(heating_scenario=heating_scenario,capacity_scenario=capacity_scenario)
nl_hydro = nl[scenario]["hydro"].sel(heating_scenario=heating_scenario,capacity_scenario=capacity_scenario)
nl_here = (nl_simple-nl_hydro).groupby("time.dayofyear")
mn = nl_here.mean(("time","member"))
std = nl_here.std(("time","member"))
mn.plot(color=pco.colors[0],label="Dispatched reservoir hydropower",linewidth=0.8)#,density=True)
axs.fill_between(mn.dayofyear.values,mn-std,mn+std,color=pco.colors[0],alpha=0.3)
axs.set_xlabel("")
axs.set_title("European mean")
axs.set_ylabel("Hydropower generation [GW]")
axs.set_xticks([60,152,244,335],labels=["Mar","Jun","Sep","Dec"])
pco.set_grid(axs)
axs.text(0.01,0.95,string.ascii_lowercase[4],weight="bold",transform=axs.transAxes)
axs.legend(loc="upper right")
plt.tight_layout()
plt.subplots_adjust(bottom=0.17)
plt.savefig("../../figs_CC_impacts/appendix_fig2_bottom_panels.png",bbox_inches="tight", dpi = 600,transparent=True)

# =================================
# === Appendix figure Panel a-d ===
# =================================

f1,ax1,f2,ax2= plot_hydro_storage(countries=["France","Norway"],heating_scenario = heating_scenario,capacity_scenario = capacity_scenario,scenario = scenario,fsize=(7,2.1))
f1.savefig("../../figs_CC_impacts/fig7_a_b.png",bbox_inches="tight", dpi = 600,transparent=True)
f2.savefig("../../figs_CC_impacts/fig7_c_d.png",bbox_inches="tight", dpi = 600,transparent=True)
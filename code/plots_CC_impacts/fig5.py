import sys
sys.path.append("../utils/")
import utils as ut
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import plot_config as pco
import string
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# =================
# === Functions ===
# =================

def get_all_nl_above_p90_atm(qu_all, nl,heating_scenario,capacity_scenario,mean_type,z500_typ="rel"):
    """
    find all net load events that are above p90 and return the atmospheric climate data that corresponds to those events
    """
    ds_temp = []
    ds_z500 = []
    lens = []
    for m,season in enumerate(["DJF","JJA"]):   
        for i,scenario in enumerate(tqdm(ut.CESM2_REALIZATION_DICT)):
            # find q90 values
            qu_transm= qu_all.sel(capacity_scenario=capacity_scenario,heating_scenario=heating_scenario)
            transmission = nl[scenario].sel(
                                heating_scenario=heating_scenario,
                                capacity_scenario=capacity_scenario
                            ).dropna(dim="time",how="all").stack(dim=("time","member"))
            transm_qu90 = transmission.where(transmission>=qu_transm,drop=True)
            
            # get seasonal anomalies of temperature and z500
            temp_season = clim_vars[scenario]["temp"].stack(dim=("time","member")).where(transm_qu90,drop=True).groupby("time.season")[season]
            z500_season = clim_vars[scenario]["z500"].stack(dim=("time","member")).where(transm_qu90,drop=True).groupby("time.season")[season]
            if mean_type == "overall":
                ds_temp.append(temp_season.mean("dim") - temps_mn_scen_ds.mean("climate").sel(season=season))
                if z500_typ=="rel":
                    ds_z500.append(z500_season.mean("dim") - z500_mn_scen_ds.mean("climate").sel(season=season))
                else:
                    ds_z500.append(z500_season.mean("dim"))
                
            elif mean_type == "per_clim":
                ds_temp.append(temp_season.mean("dim") - temps_mn_scen_ds.sel(climate=i,season=season))
                if z500_typ=="rel":
                    ds_z500.append(z500_season.mean("dim") - z500_mn_scen_ds.sel(climate=i,season=season))
                else:
                    ds_z500.append(z500_season.mean("dim"))
            lens.append(len(temp_season.dim))        
            
    ds_atm=xr.concat(ds_temp,dim="climate").to_dataset(name="temperature")
    ds_atm["Z500"]=xr.concat(ds_z500,dim="climate")
    return ds_atm, lens

# ==================
# === Open files ===
# ==================

path = "/net/xenon/climphys/lbloin/energy_boost/"
# get net load for all scenarios/climate periods, with different resolution (by technology, all together, with transmission assumption)

nl = {}
clim_vars = {}
for scenario in  ut.CESM2_REALIZATION_DICT: 
    nl[scenario] = xr.open_dataset(f"{path}net_load_transmission_{scenario}.nc").net_load
    ds_atm = xr.open_dataset(f"/net/xenon/climphys/lbloin/energy_boost/bced_atm_vars_{scenario}.nc")
    ds_z500 = xr.open_dataset(f"/net/xenon/climphys/lbloin/energy_boost/bced_z500_{scenario}.nc")
    clim_vars[scenario] = {"temp":ds_atm.temperature,"z500":ds_z500.Z500}

#list of different scenarios and assumptions
capacity_scenarios = nl[scenario].capacity_scenario.values
heating_scenarios = nl[scenario].heating_scenario.values
scenario_configs = [["current","current_electrified"],["future","fully_electrified"],["future_wind_x2","current_electrified"]]

# get 90th percentile for all climate model data combined, but separate for each scenario config
qu_all = []
for x,capacity_scenario in enumerate(capacity_scenarios):
    qu_heat = []
    for j,heating_scenario in enumerate(heating_scenarios):
        nl_for_qu = []
        for scenario in nl.keys():
            nl_for_qu.append(nl[scenario].sel(capacity_scenario=capacity_scenario,heating_scenario=heating_scenario).dropna(dim='time').values)
        qu_heat.append(np.quantile(np.array(nl_for_qu),0.9))
    qu_all.append(qu_heat)
qu_all = xr.DataArray(
    qu_all,
    dims=["capacity_scenario","heating_scenario"],
    coords = {"capacity_scenario":capacity_scenarios,"heating_scenario":heating_scenarios}
)

# find seasonal means of temperature and z500 for both climate periods
temps_mn_scen = {}
z500_mn_scen = {}
for scenario in tqdm(ut.CESM2_REALIZATION_DICT):
    temps_mn_scen[scenario] = clim_vars[scenario]["temp"].groupby("time.season").mean(("time","member"))
    z500_mn_scen[scenario] = clim_vars[scenario]["z500"].groupby("time.season").mean(("time","member"))
temps_mn_scen_ds=xr.concat([temps_mn_scen[a] for a in temps_mn_scen],dim="climate")
z500_mn_scen_ds=xr.concat([z500_mn_scen[z] for z in z500_mn_scen],dim="climate")

# =============
# === Fig 5 ===
# =============

capacity_scenario = "current"
heating_scenario = "current_electrified"
mean_type = "per_clim" # we do seasonal averages for each climate period separately

#fig specific commands
fig = plt.figure(figsize=(7.2,6))
gs = GridSpec(5, 2, height_ratios=[1,0.03,1,0.03,0.1])
proj = ccrs.PlateCarree(central_longitude=0)

# get dataset of all transmission nl values above p90
ds_atm, lens = get_all_nl_above_p90_atm(qu_all, nl,heating_scenario,capacity_scenario,mean_type,z500_typ="rel")
zmin,zmax = ut.get_minmax(ds_atm.Z500)
tmin,tmax = ut.get_minmax(ds_atm.temperature)

axes=[]
for m,season in enumerate(["DJF","JJA"]): #do for winter and summer
    to_corr_calc = ds_atm.sel(lat=slice(34, 72), lon=slice(-14, 34))
    print(season)
    print(f"Temperature correlation: {xr.corr(to_corr_calc.temperature.isel(climate=m*2),to_corr_calc.temperature.isel(climate=m*2+1)).values}")
    print(f"Z500 correlation: {xr.corr(to_corr_calc.Z500.isel(climate=m*2),to_corr_calc.Z500.isel(climate=m*2+1)).values}")
    for i,scenario in enumerate(ut.CESM2_REALIZATION_DICT):
        # make ax
        ax = fig.add_subplot(gs[m*2, i], projection=proj)
        # plot temperature
        da = ds_atm.temperature.isel(climate=m*2+i).sel(lat=slice(34, 72), lon=slice(-14, 34))
        plot_args = dict(
            ax=ax,
            x="lon", y="lat",
            transform=ccrs.PlateCarree(),
            robust=True,
            vmin=tmin,
            vmax=tmax,
            cmap="RdBu_r",
            add_colorbar = False
        )
        da.plot(**plot_args)
        # plot z500
        to_plot = ds_atm.Z500.isel(climate=m*2+i).sel(lat=slice(33,73),lon=slice(-15,35))
        g = to_plot.plot.contour(ax=ax,vmin=zmin,vmax=zmax,cmap="k",levels=[int(zmin) +x*10 for x in range(int((zmax-zmin)/10)+1)],transform=ccrs.PlateCarree())
        ax.clabel(g, inline=True, fontsize=8,levels=g.levels[::2])
        # fig config
        ax.add_feature(cfeature.BORDERS)
        ax.coastlines()
        ax.set_title(f"n={lens[m*2+i]}")
        ax.text(0.008,0.92,string.ascii_lowercase[m*2+i],weight="bold",transform=ax.transAxes)
        gl = ax.gridlines(draw_labels=True, linewidth=0,)
        gl.top_labels = False
        gl.right_labels = False
        if i > 0:
            gl.left_labels = False  # Only leftmost subplot gets y-label
        if m == 0:
            gl.bottom_labels = False  # Only leftmost subplot gets y-label
        axes.append(ax)
# add colorbar for temperature
cbar_ax = fig.add_subplot(gs[4, :])  # row below the maps, spanning both columns
im = axes[0].collections[0]  # the QuadMesh from the first plot
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar.set_label("Temperature [°C]")
# more fig configs
fig.text(0.05,0.72,"Winter",rotation='vertical',verticalalignment='center', horizontalalignment='center',fontsize=14)
fig.text(0.05,0.35,"Summer",rotation='vertical',verticalalignment='center', horizontalalignment='center',fontsize=14)
fig.text(0.3,0.95,"Historical climate",rotation='horizontal',verticalalignment='center', horizontalalignment='center',fontsize=14)
fig.text(0.72,0.95,"End-of-century climate",rotation='horizontal',verticalalignment='center', horizontalalignment='center',fontsize=14)
plt.savefig(f"../../figs_CC_impacts/fig5.png",bbox_inches="tight", dpi = 600,transparent=True)
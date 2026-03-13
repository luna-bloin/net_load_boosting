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
    tech_nl = open_joint_clim(f"{path}eng_vars_GWh","eng_vars",country_sum=True)
    mns_tech = ut.ds_hoy_in_full_time(tech_nl,"mn",dims=("time","member","climate"))
    stds_tech = ut.ds_hoy_in_full_time(tech_nl,"std",dims=("time","member","climate"))
    tech_nl =  xr.concat([tech_nl,mns_tech,stds_tech],dim=pd.Index(["full","mean","std"], name="value_type"))
    # open by country data
    country_nl = open_joint_clim(f"{path}net_load_by_country_hydro_storage","net_load_adjusted")
    region_nl = ut.get_region_mean(country_nl.drop_sel(country="Macedonia"))
    mns_region = ut.ds_hoy_in_full_time(region_nl,"mn",dims=("time","member","climate"))
    stds_region = ut.ds_hoy_in_full_time(region_nl,"std",dims=("time","member","climate"))
    region_nl =  xr.concat([region_nl,mns_region,stds_region],dim=pd.Index(["full","mean","std"], name="value_type"))
    # open storage data
    storage_nl = open_joint_clim(f"{path}net_load_hydro_storage","storage")
    mns_stor = ut.ds_hoy_in_full_time(storage_nl,"mn",dims=("time","member","climate"))
    stds_stor = ut.ds_hoy_in_full_time(storage_nl,"std",dims=("time","member","climate"))
    storage_nl =  xr.concat([storage_nl,mns_stor,stds_stor],dim=pd.Index(["full","mean","std"], name="value_type"))
    return nl, nl_qu,extremes, tech_nl, country_nl, region_nl, storage_nl

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
    to_plot_highlight = nl_boost_here.stack(event=("lead_time","member")).sel(event=event).dropna(dim="time",how="all").sel(time=slice(start,None))/1000
    to_plot_highlight = to_plot_highlight.where(to_plot_highlight > qu_parent/1000)
    to_plot_highlight = to_plot_highlight.isel(time=slice(0,to_plot_highlight.isnull().argmax("time").item()))
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

# =================================
# === Atmospheric data plotting ===
# =================================

# def plot_peak(to_plot_atm,mn_atm,tvals=0,zvals=0,wind_vals=0,save_name=[]):
#     proj = ccrs.PlateCarree(central_longitude=0)
#     #for plotting
#     variable_labels = ["Temperature anomaly [C]", "Wind speed anomaly [m/s2]"]
#     to_plot_rel = (to_plot_atm.groupby("time.dayofyear")-mn_atm).drop_vars("dayofyear")
#     if zvals==0:
#         zvals=[round(to_plot_atm.Z500.min().item(),-2),round(to_plot_atm.Z500.max().item(),-2)]
#     if tvals==0:
#         tvals = max(np.abs(to_plot_rel.temperature.min().item()),np.abs(to_plot_rel.temperature.max().item()))
#     if wind_vals==0:
#         wind_vals = max(np.abs(to_plot_rel.s_hub.min().item()),np.abs(to_plot_rel.s_hub.max().item()))
#     max_min_vals ={"temperature":[-tvals,tvals],"s_hub":[-wind_vals,wind_vals]}
#     f,ax=plt.subplots(1,2,subplot_kw={"projection": proj},figsize=(9,3.5))
#     for i,var in enumerate(["temperature","s_hub"]):
#         cmap="RdBu_r"
#         to_plot_rel[var].plot(ax=ax[i],transform=ccrs.PlateCarree(),cbar_kwargs={"shrink":0.6,"label":variable_labels[i]},cmap=cmap,vmin=max_min_vals[var][0],vmax=max_min_vals[var][1])
#         g = to_plot_atm.squeeze("time").Z500.plot.contour(ax=ax[i],vmin=zvals[0],vmax=zvals[1],cmap="k",levels=[int(zvals[0]) +x*50 for x in range(int((zvals[1]-zvals[0])/10)+1)],transform=ccrs.PlateCarree())
#         ax[i].clabel(g, inline=True, fontsize=8,levels=g.levels[::2])
#         #for plotting
#         ax[i].add_feature(cfeature.BORDERS)
#         ax[i].coastlines()
#         ax[i].set_title("")
#     f.suptitle("Date of peak net load")
#     if len(save_name) > 0:
#         f.savefig(f"../figs_tryout/atm_plot_{save_name}.png",bbox_inches="tight",transparent=True,dpi=600)
#     return tvals,zvals,wind_vals

# def parent_atm_processing(date_ranges,member,season,atm_ds,z500_ds,mn_atm_ds,mn_z500_ds=[]):
#     """returns xr datasets of seasonal anomalies for a given extreme period, separated into three stages of the extreme (temporal mean)"""
#     parent_atm = []
#     parent_z500 = []
#     for date_range in date_ranges:
#         # open, select date and member, mean over time, then remove seasonal mean
#         parent_atm.append(atm_ds.sel(member=member,time=date_range).mean("time") - mn_atm_ds.sel(season=season))
#         if len(mn_z500_ds) > 0:
#             parent_z500.append(z500_ds.sel(member=member,time=date_range).mean("time") - mn_z500_ds.sel(season=season))
#         else:
#             parent_z500.append(z500_ds.sel(member=member,time=date_range).mean("time"))# - mn_z500_ds.sel(season=season))
#     # concatenate into data set
#     parent_atm = xr.concat(parent_atm,dim=pd.Index([str(date_range.start)[0:10] for date_range in date_ranges], name="date_ranges"))
#     parent_z500 = xr.concat(parent_z500,dim=pd.Index([str(date_range.start)[0:10] for date_range in date_ranges], name="date_ranges"))
#     return parent_atm, parent_z500

# def plot_weather_anom(parent_atm,parent_z500):
#     """plots maps of climate variables during an extreme event"""
#     proj = ccrs.PlateCarree(central_longitude=0)
#     #for plotting
#     variable_labels = ["Temperature anomaly [C]", "Wind speed anomaly [m/s2]","Solar radiation anomaly [W/m2]"]
#     stages = ["Beginning", "Middle", "End"]
#     zmin = (parent_z500).Z500.min()
#     zmax = (parent_z500).Z500.max()
    
#     for i,var in enumerate(["temperature","s_hub","global_horizontal"]):
#         #plot each variable in separate figure as map
#         f = parent_atm[var].plot(x="lon",y="lat",col="date_ranges",transform=ccrs.PlateCarree(),cbar_kwargs={"shrink":0.65,"label":variable_labels[i]},cmap="RdBu_r",subplot_kws={"projection": proj})
#         for j, ax in enumerate(f.axes.flat):
#             if var == "s_hub":
#                 # plot z500 anomalies on top of wind speed
#                 date = parent_atm.date_ranges.values[j]
#                 g = parent_z500.Z500.sel(date_ranges=date).plot.contour(ax=ax,vmin=zmin,vmax=zmax,cmap="k",levels=[int(zmin) +x*50 for x in range(int((zmax-zmin)/10)+1)],transform=ccrs.PlateCarree())
#                 ax.clabel(g, inline=True, fontsize=8,levels=g.levels[::2])
#             #for plotting
#             ax.add_feature(cfeature.BORDERS)
#             ax.coastlines()
#             ax.set_title(stages[j])
#     return f,ax

# =====================
# === Tech plotting ===
# =====================

# def plot_event_overview(cum_parent,dur_parent,cum,dur,qu,nl_one_scen,f,ax,color,plot_relative_time=True,bottom=True,boost_end=False):
#     #ax[0] shows the event net load as it unfolds
#     nl_top = find_nl_top(nl_one_scen, cum_parent,dur,boost_end=boost_end)/1000
#     if plot_relative_time == True:
#         time_rel = (nl_top.time - nl_top.time[0])
#         time_rel = time_rel/(time_rel[1]*24)
#         ax[0].plot(time_rel,nl_top,color=color)
#         if bottom ==True:
#             ax[0].set_xlabel("Days since beginning of event")
#     else:
#         nl_top.plot(ax=ax[0],color=color)
#         ax[0].set_xlabel("")
#     ax[0].axhline(qu/1000,color="k",linestyle="--")
#     pco.set_grid(ax[0])
#     if bottom == False:
#         ax[0].set_title(f"Temporal evolution of extreme")
#     ax[0].set_ylabel("Net load [TW]")
    
    
#     # ax[1] shows cumulative threshold exceedance
#     sns.violinplot(cum,ax=ax[1],inner="point",alpha=0.3,color=pco.colors[2])
#     ax[1].plot(cum_parent,"D",markersize=3.5,zorder=4,color=color,mec="k",mew=0.7,label= f"{cum_parent.member.item()}-{cum_parent.time.dt.year.item()}-{cum_parent.time.dt.month.item():02d}")
#     ax[1].set_ylabel("Exceedance [TWh]")
#     ax[1].set_ylim(0,cum.max()+10)
    
#     # ax[2] shows duration
#     sns.violinplot(dur,ax=ax[2],inner="point",alpha=0.3,color=pco.colors[2])
#     ax[2].plot(dur_parent,"D",markersize=3.5,zorder=4,color=color,mec="k",mew=0.7)
#     ax[2].set_ylabel("Duration [days]")
#     ax[2].set_ylim(0,dur.max()+2)
#     if bottom == False:
#         ax[2].set_title("Comparison statistics")
#     # ax[3] shows seasonality
#     sns.violinplot(cum.time.dt.dayofyear,ax=ax[3],inner="point",alpha=0.3,color=pco.colors[2])
#     ax[3].plot(cum_parent.time.dt.dayofyear,"D",markersize=3.5,zorder=4,color=color,mec="k",mew=0.7)
#     ax[3].set_ylim(0,365)
#     ax[3].set_yticks([60,152,244,335],labels=["Mar","Jun","Sep","Dec"])
#     ax[3].set_ylabel("Seasonality")
#     # plotting configs
#     plt.tight_layout()

# def plot_one_event(tech_nl,region_nl,nl_one_scen,top_event,dur_event,dur_scenario,atm_ds,atm_mn,boost_tech=[],boost_region=[],boost_min_dur=[],boost_atm_ds=[],boost_peak_date=[],save_name=[]):
#     #find beginning, duration and end of event
#     end = top_event.time.item()
#     start = end - timedelta(days=int(dur_event)+1)
#     if len(boost_tech)>0:
#         boost_end = start + timedelta(days=boost_min_dur+1) # keep plotting parent and mn std dev for duration of boosted event
#         end_time = boost_end +timedelta(days=7) #boost_tech.dropna(dim="time").time[-int(24*7/2)].item() # remove half a week at  the end bc of rolling mean
#         if end_time < end:
#             end_time = end
#     else:
#         end_time = end
#         boost_end = []

#     #find date of peak net load
#     peak_date = str(nl_one_scen.sel(time=slice(start, end)).idxmax().item())[0:10]
#     # plot tech and region breakdown plots
#     for i,to_plot_all in enumerate([tech_nl,region_nl]):
#         # weekly rolling mean for visual clarity
#         to_plot = to_plot_all.sel(value_type="full").sel(time=slice(start,end_time))/1000 #.rolling(time=24*7,center=True).mean()
#         to_plot_all = to_plot_all.sel(time=slice(start,end_time))/1000
#         if i ==0:
#             # boosted runs
#             if len(boost_tech) > 0:
#                 #boost_tech = boost_tech.rolling(time=24*7,center=True).mean()/1000
#                 boost_tech = find_nl_top(boost_tech,top_event,dur_scenario,boost=True).sel(time=slice(start,end_time))/1000
#             f,ax = plot_by_tech(to_plot,to_plot_all.sel(value_type="mean"),to_plot_all.sel(value_type="std"),start,end,boost=boost_tech,boost_end = boost_end)
#             if len(save_name) > 0:
#                 f.savefig(f"../figs_tryout/tech_breakdown_{save_name}.png",bbox_inches="tight",transparent=True,dpi=600)
#         if i ==1:
#             # boosted runs
#             if len(boost_region) > 0:
#                 #boost_region = boost_region.rolling(time=24*7,center=True).mean()/1000
#                 boost_region = find_nl_top(boost_region,top_event,dur_scenario,boost=True).sel(time=slice(start,end_time))/1000
#             f,ax = plot_by_region(to_plot,to_plot_all.sel(value_type="mean"),to_plot_all.sel(value_type="std"),start,end,boost=boost_region,boost_end = boost_end)
#             if len(save_name) > 0:
#                 f.savefig(f"../figs_tryout/region_breakdown_{save_name}.png",bbox_inches="tight",transparent=True,dpi=600)
#     # plot atmospheric plots
#     tvals,zvals,wind_vals = plot_peak(atm_ds.sel(time=peak_date),atm_mn,save_name=save_name)
#     return tvals,zvals,wind_vals

# def plot_by_tech(to_plot,mn_here,st_here,event_start,event_end,boost=[],boost_end = [],f=[],axs=[],plot_mn_std=True,linewidth=1.4):
#     # fig config
#     if len(f) == 0:
#         f,axs=plt.subplots(2,5,figsize=(15,5),sharex=True,sharey=True,gridspec_kw={'width_ratios': [1,0.02,1,1,1]})
#     axs[0][1].axis('off')
#     axs[1][1].axis('off')
#     techs = [['heating-demand','cooling-demand'],['Wind_onshore','Wind_offshore'], ['hydro_inflow', 'hydro_ror'],['PV', ""]]
#     colors = [[pco.colors[3],pco.colors[4]],[pco.colors[1],pco.colors[2]], [pco.colors[5], pco.colors[6]],[pco.colors[0], ""] ]
    
#     for i, tech in enumerate(techs):
#         for j,technology in enumerate(tech):
#             #skip ax[1], which exists to draw a line between demand and generation
#             ax = axs[j][i+1*(math.ceil(i/5))]
#             if technology == "":
#                 ax.axis('off')
#                 continue
#             to_plot_tech = to_plot.sel(technology=technology)
#             if len(boost) >0:
#                 boost_tech = boost.sel(technology=technology)
#             #plot mean + std dev
#             mn_plot = mn_here.sel(technology=technology)
#             st_plot = st_here.sel(technology=technology)
#             if plot_mn_std == True:
#                 mn_plot.plot(ax=ax,color=colors[i][j])
#                 upper = mn_plot + st_plot
#                 lower = mn_plot - st_plot
#                 ax.fill_between(upper["time"].values, upper,lower,alpha=0.3, color=colors[i][j])
#             # plot event
#             to_plot_tech.plot(ax=ax,color=colors[i][j],linewidth=linewidth)
#             if len(boost) >0:
#                 for event in boost_tech.event:
#                     boost_tech.sel(event=event).plot(ax=ax,color=colors[i][j],alpha=0.5,linewidth=0.6)
#             #highlight time frame of event
#             ax.axvspan(event_start,event_end,color="k",alpha=0.1)
#             if len(boost) >0:
#                 ax.axvspan(event_end,boost_end,color="k",alpha=0.05)
#             # plot configs
#             ax.set_title(ut.tech_dict[technology])
#             if i ==0:
#                 ax.set_ylabel("Demand [TW]")
#             elif i ==1 :
#                 ax.set_ylabel("Generation [TW]")
#             else:
#                 ax.set_ylabel("")
#             ax.set_xlabel("")
#             pco.set_grid(ax)
#     f.add_artist(Line2D([0.277, 0.277], [0.01, 0.99],
#                           transform=f.transFigure,
#                         linewidth=1.3,
#                           color="black"))
#     plt.tight_layout()
#     return f,axs

# def plot_by_tech_anom(to_plot,f=None,axs=None,xlims=[],ylims=[]):
#     # fig config
#     if axs is None:
#         f,axs=plt.subplots(2,5,figsize=(15,5),sharex=True,sharey=True,gridspec_kw={'width_ratios': [1,0.02,1,1,1]})
#     axs[0][1].axis('off')
#     axs[1][1].axis('off')
#     techs = [['heating-demand','cooling-demand'],['Wind_onshore','Wind_offshore'], ['hydro_inflow', 'hydro_ror'],['PV', ""]]
#     colors = [[pco.colors[3],pco.colors[4]],[pco.colors[1],pco.colors[2]], [pco.colors[5], pco.colors[6]],[pco.colors[0], ""] ]
#     #get relative time
#     time_rel = (to_plot.time - to_plot.time[0])
#     time_rel = time_rel/(time_rel[1]*24)    
#     to_plot["time_rel"] = time_rel
#     to_plot=to_plot.swap_dims({"time": "time_rel"})
#     for i, tech in enumerate(techs):
#         for j,technology in enumerate(tech):
#             #skip ax[1], which exists to draw a line between demand and generation
#             ax = axs[j][i+1*(math.ceil(i/5))]
#             if technology == "":
#                 ax.axis('off')
#                 continue
#             to_plot_tech = to_plot.sel(technology=technology)
#             # plot event
#             to_plot_tech.plot(ax=ax,color=colors[i][j])
#             # plot configs
#             ax.set_title(ut.tech_dict[technology])
#             if i ==0:
#                 ax.set_ylabel("Demand"+"\n"+"Anomaly [TW]")
#             elif i ==1 :
#                 ax.set_ylabel("Generation"+"\n"+"Anomaly [TW]")
#             else:
#                 ax.set_ylabel("")
#             ax.set_xlabel("Time [days]")
#             pco.set_grid(ax)
#             ax.axhline(0,linestyle="--",color="k")
#             if len(xlims)!=0:
#                 ax.set_xlim(xlims[0],xlims[1])
#             if len(ylims)!=0:
#                 ax.set_ylim(ylims[0],ylims[1])
#     f.add_artist(Line2D([0.277, 0.277], [0.01, 0.99],
#                           transform=f.transFigure,
#                         linewidth=1.3,
#                           color="black"))
#     plt.tight_layout()
#     return f,axs

# def plot_by_region(to_plot,mn_here,st_here,event_start,event_end,boost=[],boost_end = []):
#     f,ax=plt.subplots(1,4,figsize=(15,3),sharey=True)
#     regions = ["Northern Europe", "Southern Europe", "Western Europe","Eastern Europe"]
#     for j,region in enumerate(regions):
#         to_plot_region = to_plot.sel(region=region)
#         if len(boost) > 0:
#             boost_region = boost.sel(region=region)
#         #plot mn +- std dev
#         mn_region = mn_here.sel(region=region)
#         st_region = st_here.sel(region=region)
#         mn_region.plot(ax=ax[j],color=pco.colors[j])
#         upper = mn_region + st_region
#         lower = mn_region - st_region
#         ax[j].fill_between(upper["time"].values, upper,lower,alpha=0.3, color=pco.colors[j])
#         #plot event
#         to_plot_region.plot(ax=ax[j],color=pco.colors[j],linewidth=1.4)
#         if len(boost) > 0:
#             for event in boost_region.event:
#                     boost_region.sel(event=event).plot(ax=ax[j],color=pco.colors[j],alpha=0.5,linewidth=0.6)
#         #highlight time frame of event
#         ax[j].axvspan(event_start,event_end,color="k",alpha=0.1)
#         if len(boost) >0:
#             ax[j].axvspan(event_end,boost_end,color="k",alpha=0.05)
#         #plot configs
#         ax[j].set_title(region)
#         if j == 0:
#             ax[j].set_ylabel("Net load [TW]")
#         else:
#             ax[j].set_ylabel("")
#         ax[j].set_xlabel("")
#         pco.set_grid(ax[j])
#     plt.tight_layout() 
#     return f,ax


# def plot_by_region_anom(to_plot,f=None,ax=None,xlims=[],ylims=[]):
#     # fig config
#     if ax is None:
#         f,ax=plt.subplots(1,4,figsize=(15,3),sharey=True)
#     regions = ["Northern Europe", "Southern Europe", "Western Europe","Eastern Europe"]
#     #get relative time
#     time_rel = (to_plot.time - to_plot.time[0])
#     time_rel = time_rel/(time_rel[1]*24)    
#     to_plot["time_rel"] = time_rel
#     to_plot=to_plot.swap_dims({"time": "time_rel"})
#     for j,region in enumerate(regions):
#         to_plot_region = to_plot.sel(region=region)
#         # plot event
#         to_plot_region.plot(ax=ax[j],color=pco.colors[j])
#         # plot configs
#         ax[j].set_title(region)
#         if j == 0:
#             ax[j].set_ylabel("Net load anomaly [TW]")
#         else:
#             ax[j].set_ylabel("")
#         ax[j].set_xlabel("Time [days]")
#         pco.set_grid(ax[j])
#         ax[j].axhline(0,linestyle="--",color="k")
#         if len(xlims)!=0:
#             ax[j].set_xlim(xlims[0],xlims[1])
#         if len(ylims)!=0:
#             ax[j].set_ylim(ylims[0],ylims[1])
#     plt.tight_layout()
#     return f,ax
import sys
sys.path.append("../utils/")
# local imports
import utils as ut
import plot_config as pco
import extreme_analysis as exa
#standard
import xarray as xr
import numpy as np
#plotting
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
#other 
from datetime import timedelta
import string


# =================
# === Functions ===
# =================

def plot_main_range(to_plot,ax,color,title,lab = "_no",ylabel="",ylim=[]):
    """plots the full range of events, as well as the first (most extreme) event"""
    #plot
    ax.fill_between(to_plot.time_rel,to_plot.max("event"),to_plot.min("event"),alpha=0.3,color=color)
    to_plot.isel(event=0).plot(color=color,ax=ax,label=lab)
    #plot config
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if len(ylim) >0:
        ax.set_ylim(ylim)
    return ax

def get_top_5_rel(ds_nl,cum,dur,heat,capac,climate,rel_time=True,percentage=False,rel_vals=True):
    """Get top 5 energy shortfall events (given a heating scenario heat and capacity scenario capac in a given climate), 
    and open them in ds_nl (can be technological values, regional net load...) and adds them into one dataset. If rel_time 
    is true, it counts time as days since the event started. If percentage is true, it gives values as anomaly %, if rel_vals 
    is true, it gives values as hour of year anomalies."""
    to_plots = []
    for i in range(5):
        top_event = cum[i]
        dur_event = dur.sel(event = top_event.event.item())
        mem = top_event.member.item()
        ds_here = ds_nl.sel(heating_scenario=heat,capacity_scenario=capac,climate=climate,member=mem).sel(time=slice(top_event.time.item() - timedelta(days=int(dur_event)),top_event.time.item()))/1000
        if rel_vals ==  True:
            to_plot = ds_here.sel(value_type="full") - ds_here.sel(value_type="mean")
        else:
            to_plot = ds_here.sel(value_type="full")
        if percentage == True:
            to_plot = to_plot*100/ds_here.sel(value_type="mean")
        if rel_time == True:
            time_rel = (to_plot.time - to_plot.time[0])
            time_rel = time_rel/(time_rel[1]*24)    
            to_plot["time_rel"] = time_rel
            to_plot=to_plot.swap_dims({"time": "time_rel"})
        to_plots.append(to_plot)
    return xr.concat(to_plots,dim="event")

def plot_tech_region(top5,ylim_tech = [-0.62,0.6],ylim_region = [-0.025,0.075],ylim_storage=[-10,10],anom=True):
    """plots top5 events in terms of demand, generation, hydropower storage and regional net load"""
    f,axs=plt.subplots(2,4,figsize=(10,4.5),sharex=True)
    if anom ==True:
        anom_lab = " anomaly"
    else:
        anom_lab = ""
    
    # plot heating and cooling demand
    demands = {"heating-demand":"Heating","cooling-demand":"Cooling"}
    color = [pco.colors[3],pco.colors[4]]
    for i, typ in enumerate(demands):
        to_plot = top5["tech"].sel(technology=typ)
        #plot range of top5
        axs[0][0] = plot_main_range(to_plot,axs[0][0],color[i],"Demand",lab=demands[typ],ylabel=f"Demand{anom_lab} [TW]",ylim=ylim_tech)
        axs[0][0].legend()
    
    # plot on+offshore wind
    to_plot = top5["tech"].sel(technology=["Wind_onshore","Wind_offshore"]).sum("technology", skipna=False)
    axs[0][1] = plot_main_range(to_plot,axs[0][1],"cadetblue","On- and offshore wind",ylabel=f"Generation{anom_lab} [TW]",ylim=ylim_tech)
    
    # plot solar PV
    to_plot = top5["tech"].sel(technology="PV")
    axs[0][2] = plot_main_range(to_plot,axs[0][2],pco.colors[0],"Solar PV",ylim=ylim_tech)
    
    # plot hydro storage
    if anom == True:
        ylab = f"Storage{anom_lab} [%]"
    else:
        ylab = f"Storage{anom_lab} [TWh]"
    axs[0][3] = plot_main_range(top5["storage"],axs[0][3],pco.colors[6],"Hydropower storage",ylabel=ylab,ylim=ylim_storage)
    
    # plot regional net load
    regions = ["Northern Europe", "Southern Europe", "Western Europe","Eastern Europe"]
    for j,region in enumerate(regions):
        to_plot = top5["region"].sel(region=region)
        if j==0:
            lab = f"Net load{anom_lab} [TW]"
        else:
            lab = ""
        axs[1][j] = plot_main_range(to_plot,axs[1][j],"k",region,ylabel=lab,ylim=ylim_region)
    
    # General figure configs
    for i,a in enumerate(axs.flatten()):
        pco.set_grid(a)
        if anom == True:
            a.axhline(0,linestyle="--",color="k") # plot 0 line
        if i < 4:
            a.set_xlabel("")
        else:
            a.set_xlabel("Time [days]")
        if i%4==0 or i ==3:
            continue
        a.set_yticklabels([])
    f.tight_layout()
    return f,axs

def plot_waether_maps(to_plot_atm,mn_atm,tvals=0,zvals=0,wind_vals=0):
    """plots maps of temperature and wind speed anomalies, with z500 contouring overlaid. 
    tvals,wind_vals and zvals can be set to create the max min values of the mape"""
    proj = ccrs.PlateCarree(central_longitude=0)
    #for plotting
    variable_labels = ["Temperature anomaly [C]", "Wind speed anomaly [m/s]"]
    to_plot_rel = (to_plot_atm.groupby("time.dayofyear")-mn_atm).mean("time")
    to_plot_atm = to_plot_atm.mean("time")
    if zvals==0:
        zvals=[round(to_plot_atm.Z500.min().item(),-2),round(to_plot_atm.Z500.max().item(),-2)]
    if tvals==0:
        tvals = max(np.abs(to_plot_rel.temperature.min().item()),np.abs(to_plot_rel.temperature.max().item()))
    if wind_vals==0:
        wind_vals = max(np.abs(to_plot_rel.s_hub.min().item()),np.abs(to_plot_rel.s_hub.max().item()))
    max_min_vals ={"temperature":[-tvals,tvals],"s_hub":[-wind_vals,wind_vals]}
    f,ax=plt.subplots(1,2,subplot_kw={"projection": proj}, figsize=(9, 3))
    cmap=["RdBu_r","PuOr"]
    for i,var in enumerate(["temperature","s_hub"]):
        
        p = to_plot_rel[var].plot(ax=ax[i],transform=ccrs.PlateCarree(),cbar_kwargs={"shrink":0.85,"location": "left","label": variable_labels[i],"fraction":0.07},cmap=cmap[i],vmin=max_min_vals[var][0],vmax=max_min_vals[var][1])
        g = to_plot_atm.Z500.plot.contour(ax=ax[i],vmin=zvals[0],vmax=zvals[1],cmap="k",levels=[int(zvals[0]) +x*50 for x in range(int((zvals[1]-zvals[0])/10)+1)],transform=ccrs.PlateCarree())
        ax[i].clabel(g, inline=True, fontsize=8,levels=g.levels)#[::2])
        #for plotting
        ax[i].add_feature(cfeature.BORDERS)
        ax[i].coastlines()
        ax[i].set_title("")        
    return f,ax,tvals,zvals,wind_vals

def plot_tech_region_weather(heat,capac,climate,rel_vals=True,ylim_tech = [-0.9,0.6],ylim_region = [-0.025,0.085],ylim_storage=[-12,9],yticks=True,tvals=[],zvals=[],wind_vals=[]):
    """plots the technological, regional and weather breakdown of the top 5 energy shortfall events in a given heating and capacity scenario for a given climate"""
    # === get net load and top events for given heating, capacity scenario and climate period ===
    nl_one_scen = nl.sel(heating_scenario=heat,capacity_scenario=capac,climate=climate)
    qu = nl_qu.sel(capacity_scenario=capac,heating_scenario=heat)
    extreme_here = extremes.sel(heating_scenario=heat,capacity_scenario=capac,climate=climate).dropna(dim="event",how="all")
    dur = extreme_here.sel(typ = "dur")
    cum = extreme_here.sel(typ="cum")
    cum = cum.sortby(cum,ascending=False)
    
    # === Get top 5 events in terms of tech, storage and regional nl ===
    top5 = {}
    nl_dict = {"tech": tech_nl,"region": region_nl,"storage": storage_nl}
    for ds_nl in nl_dict:
        if ds_nl == "storage":
            percentage =True
        else:
            percentage=False
        top5[ds_nl] = get_top_5_rel(nl_dict[ds_nl],cum,dur,heat,capac,climate,percentage=percentage,rel_vals=rel_vals)
    
    # === Figure ===
    f,ax = plot_tech_region(top5,ylim_tech = ylim_tech,ylim_region = ylim_region,ylim_storage=ylim_storage,anom=rel_vals)
    for i,a in enumerate(ax.flatten()):
        if rel_vals == True:
            x = i+2
        else:
            x=i
        a.text(0.01,0.91,string.ascii_lowercase[x],weight="bold",transform=a.transAxes)
        if i > 3 and yticks==True:
            a.set_yticks([0,0.03,0.06])
    f.tight_layout()
    if rel_vals == False:
        add_save="_abs_vals"
    else:
        add_save=""
    f.savefig(f"../../figs_storyline_extremes/tech_region_breakdown{add_save}_top5_{capac}_{heat}_{climate}.png",bbox_inches="tight", transparent=True, dpi=600)

    if rel_vals == True:
        # get atmospheric map for peak net load in each of the top 5 events
        atms = []
        for top_event in cum[0:5]:
            dur_event = dur.sel(event = top_event.event.item())
            mem = top_event.member.item()
            nl_here = nl_one_scen.sel(member=mem,time=slice(top_event.time.item()-timedelta(dur_event.item()),top_event.time.item()))
            peak_date = nl_here.idxmax(dim="time").item()
            atm_here=atm[climate].sel(member=mem,time=str(peak_date)[0:10])
            atms.append(atm_here) 
        atms_mn_top5 = xr.concat(atms,dim="event").mean("event")
    
        if len(zvals)>0:
            f,ax,tvals,zvals,wind_vals = plot_waether_maps(atms_mn_top5,atm_mn,tvals=tvals,zvals=zvals,wind_vals=wind_vals)
        else:
            f,ax,tvals,zvals,wind_vals = plot_waether_maps(atms_mn_top5,atm_mn)
        
        for i,a in enumerate(ax):
            a.text(0.01,0.95,string.ascii_lowercase[i],weight="bold",transform=a.transAxes)
        f.tight_layout()
        f.savefig(f"../../figs_storyline_extremes/atm_plot_top5_{capac}_{heat}_{climate}.png",bbox_inches="tight",transparent=True,dpi=600)
    return tvals,zvals,wind_vals


# =====================
# === Opening files ===
# =====================
print("opening files")
path = "/net/xenon/climphys/lbloin/energy_boost/"
# open all net load types for parent events
nl, nl_qu,extremes, tech_nl,region_nl,storage_nl = exa.open_all_parent_nl(path)
# open atmospheric variables
atm,atm_mn = exa.open_atm_vars(path)
print("files opened")

# ================
# === Figure 2 ===
# ================
print("plotting fig 2-3")
heat = "fully_electrified"
capac = "future"
climate = 'SSP370'
tvals,zvals,wind_vals = plot_tech_region_weather(heat,capac,climate)


# ================
# === Figure 3 ===
# ================

heat = "current_electrified"
capac = "future_wind_x2"
climate = 'SSP370'
plot_tech_region_weather(heat,capac,climate,tvals=tvals,zvals=zvals,wind_vals=wind_vals)
print("fig 2-3 saved")


# ============================
# === Appendix figures 4-5 ===
# ============================

print("plotting appendix fig 4-8")
heat = "fully_electrified"
capac = "future"
climate = 'historical'
plot_tech_region_weather(heat,capac,climate,ylim_tech = [-0.7,0.9],ylim_region = [-0.025,0.09],ylim_storage=[-35,5],yticks=False)

heat = "current_electrified"
capac = "future_wind_x2"
climate = 'historical'
plot_tech_region_weather(heat,capac,climate,ylim_tech = [-0.8,0.1],ylim_region = [-0.025,0.085],ylim_storage=[-7,5],yticks=False)


# ===========================
# === Appendix figure 6-7 ===
# ===========================

heat = "fully_electrified"
capac = "future"
climate = 'SSP370'
plot_tech_region_weather(heat,capac,climate,rel_vals=False,ylim_tech = [0,1.3],ylim_region = [-0.06,0.08],ylim_storage=[80,110],yticks=False)

heat = "current_electrified"
capac = "future_wind_x2"
climate = 'SSP370'
plot_tech_region_weather(heat,capac,climate,rel_vals=False,ylim_tech = [0,1.3],ylim_region = [-0.06,0.08],ylim_storage=[80,110],yticks=False)

# =========================
# === Appendix figure 8 ===
# =========================

heat = "fully_electrified"
capac = "future"
climate = 'SSP370'
# === get net load and top events for given heating, capacity scenario and climate period ===
nl_one_scen = nl.sel(heating_scenario=heat,capacity_scenario=capac,climate=climate)
qu = nl_qu.sel(capacity_scenario=capac,heating_scenario=heat)
extreme_here = extremes.sel(heating_scenario=heat,capacity_scenario=capac,climate=climate).dropna(dim="event",how="all")
dur = extreme_here.sel(typ = "dur")
cum = extreme_here.sel(typ="cum")
cum = cum.sortby(cum,ascending=False)

# === Get top 5 events in terms of tech, storage and regional nl ===
top5 = get_top_5_rel(tech_nl,cum,dur,heat,capac,climate)

f,ax=plt.subplots(1,2,figsize=(7.2,3),sharex=True,sharey=True)
#plot the two types of hydropower
for i,tech in enumerate(["hydro_ror","hydro_inflow"]):
    ax[i] = plot_main_range(top5.sel(technology=tech),ax[i],pco.colors[4+i],ut.generation_dict[tech])
# General figure configs
for i,a in enumerate(ax):
    pco.set_grid(a)
    a.axhline(0,linestyle="--",color="k") # plot 0 line
    a.set_xlabel("Time [days]")
    a.text(0.01,0.91,string.ascii_lowercase[i],weight="bold",transform=a.transAxes)
ax[0].set_ylabel("Generation anomaly [TW]")
f.tight_layout()  
f.savefig(f"../../figs_storyline_extremes/hydro_top5_{capac}_{heat}_{climate}.png",bbox_inches="tight",transparent=True,dpi=600)
print("appendix fig 4-8 saved")

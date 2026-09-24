import sys
sys.path.append("../utils/")
# local imports
import utils as ut
import plot_config as pco
import extreme_analysis as exa
#plotting
import matplotlib.pyplot as plt
import seaborn as sns
#other 
import xarray as xr
import pandas as pd
import numpy as np
from datetime import timedelta
import string
import matplotlib.dates as mdates
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# =================
# === Functions ===
# =================

def open_all_necessary_files_for_boosting(path,boost_dates,start_parent,scenario,member,heat,capac):
    """Opens all necessary data for doing the three types of boosting plot"""
    # open boosted files 
    nl_boost = exa.open_boost(path,boost_dates,start_parent,scenario,member,typ="transmission").sel(heating_scenario=heat,capacity_scenario=capac)
    tech_boost = exa.open_boost(path,boost_dates,start_parent,scenario,member,typ="eng_vars").sel(heating_scenario=heat,capacity_scenario=capac)
    storage_boost = exa.open_boost(path,boost_dates,start_parent,scenario,member,typ="storage").sel(heating_scenario=heat,capacity_scenario=capac)
    # find net load in correct scenario
    nl_parent = nl.sel(heating_scenario=heat,capacity_scenario=capac,climate=climate)
    qu_parent = nl_qu.sel(heating_scenario=heat,capacity_scenario=capac)
    # find all extremes in this scenario
    extreme_scenario = extremes.sel(heating_scenario=heat,capacity_scenario=capac,climate=climate).dropna(dim="event",how="all")
    dur_scenario = extreme_scenario.sel(typ="dur")
    cum_scenario = extreme_scenario.sel(typ="cum")
    cum_scenario = cum_scenario.sortby(cum_scenario,ascending=False)
    # find net load of parent event (rough time horizon for event)
    nl_parent_only_event = nl_parent.sel(member=member,time=slice(start_parent,ut.str_to_cftime_noleap(start_parent) + timedelta(days=80)))
    cum_parent, dur_parent = exa.spa_algo(nl_parent_only_event, qu_parent)
    cum_parent = cum_parent.where(cum_parent==cum_parent.max(),drop=True)/1000 # this is the parent of the boosted simulation
    dur_parent = dur_parent.where(dur_parent.time==cum_parent.time,drop=True)/24
    end = cum_parent.time.item()
    start = end - timedelta(days=round(dur_parent.item()))
    #find boosted energy shortfall events
    dur_cum_boost = exa.find_dur_cum_boost(nl_boost,qu_parent)
    #find top and bottom 5
    ds = dur_cum_boost.stack(event=("lead_time","member"))
    top = ds.sortby(ds.sel(typ="cum"),ascending=False).isel(event=0)
    bottom = ds.sortby(ds.sel(typ="cum"),ascending=False).isel(event=-1)
    return nl_boost,tech_boost, storage_boost, qu_parent, nl_parent_only_event,cum_parent, start, end, dur_cum_boost, top,bottom,cum_scenario

def plot_mn_std(to_plot,ax,title):
    to_plot.sel(value_type="mean").plot(color="k",ax=ax,linewidth=0.5)
    ax.fill_between(to_plot.time.values,to_plot.sel(value_type="mean") - to_plot.sel(value_type="std"),to_plot.sel(value_type="mean") + to_plot.sel(value_type="std"),color="k",alpha=0.2,label="Seasonal mean $\pm$ standard deviation")
    ax.set_title(title)

def get_on_offshore_wind(ds):
    return ds.sel(technology=["Wind_onshore","Wind_offshore"]).sum("technology", skipna=False)

def plot_solid_dotted(to_plot_solid,to_plot_dotted,color,ax,end,lw=[1,0.7],lst=["solid","dotted"],label="_no",cum="False"):
    #plot solid for event only
    to_plot_solid.plot(
        color=color,
        linewidth=lw[0],
        linestyle = lst[0],
        label=label,
        zorder=4,
        ax=ax
    ) 
    if cum == "False":
        # plot before and after event (dotted line)
        to_plot_dotted.plot(
            color=color,
            linewidth=lw[1],
            linestyle = lst[1],
            zorder=3,
            ax=ax
        ) 
    else:
        ax.hlines(
            y=to_plot_dotted,
            xmin=end,
            xmax=end+timedelta(days=50),
            color=color,
            linewidth=lw[1],
            linestyle = lst[1],
        ) 

def open_atm_var_snapshots(var,times,top,bottom):
    """opens 2d maps of a given variable var of the parent and worst and best boosted simulation for a list of days. returns as one dataset"""
    ds_parent = xr.open_dataset(f"/net/xenon/climphys/lbloin/energy_boost/bced_{var}_SSP370_A.nc")[var]
    ds = []
    ds.append(ds_parent.sel(time=times))
    for event in [top,bottom]:
        ds.append(xr.open_dataset(f"/net/xenon/climphys/lbloin/energy_boost/bced_{var}_SSP370_boost_A_{event.lead_time.item()}.nc")[var].sel(time=times,member=event.member).drop_vars(["member","capacity_scenario","heating_scenario","quantile","event","lead_time"]))
    ds = xr.concat(ds, dim=pd.Index(["Parent","Most extreme","Least extreme"],name="event"))
    return ds

def floor_to_day(t):
    return t.replace(hour=0, minute=0, second=0, microsecond=0)

def plot_boosting_overview(nl_boost,nl_parent_only_event,start,end,qu_parent,cum_parent,top,bottom,parent_tech,parent_storage,tech_boost,storage_boost,save_info,ylim=[0.05,1.45],xlim=23):
    fig = plt.figure(figsize=(8, 7.2))
    gs = fig.add_gridspec(nrows=4,ncols=2,width_ratios=[1.8, 1],)
    # Left column: 2 plots, each spanning 2 rows
    ax1 = fig.add_subplot(gs[0:2, 0])
    ax2 = fig.add_subplot(gs[2:4, 0], sharex=ax1)
    # Right column
    ax3 = fig.add_subplot(gs[0, 1], sharex=ax1)
    ax4 = fig.add_subplot(gs[1, 1], sharex=ax1, sharey=ax3)
    ax5 = fig.add_subplot(gs[2, 1], sharex=ax1, sharey=ax3)
    ax6 = fig.add_subplot(gs[3, 1], sharex=ax1)
    
    lws = [1,0.7]
    lst = ["solid","dotted"]
    colors = ["k",pco.colors[0],"sienna","olive"]
    
    # ==================================================
    # === plot instantaneous and cumulative net load ===
    # ==================================================
    # all boosted runs in correct format
    nl_boost_right_format = nl_boost.stack(event=("lead_time","member")).dropna(dim="time",how="all")
    # find cumulative net load for all boosted runs
    cumsum_boosts = exa.cumulative_nl_boost(nl_boost,start,qu_parent)
    
    # === plot parent net load ====
    drought = (nl_parent_only_event/1000).sel(time=slice(start,end)) #nl for event only 
    #instantaneous
    plot_solid_dotted(drought,(nl_parent_only_event/1000),colors[0],ax1,end,label="Parent event")
    #plot cumulative threshold exceedence 
    cum_here = (drought-qu_parent/1000).where(drought - qu_parent/1000 > 0,0).cumsum()
    plot_solid_dotted(cum_here,cum_parent,colors[0],ax2,end,cum="True")
    
    # === plot top and bottom event of boosted net load ===
    lab_typ = ["Most", "Least"]
    for i,top_bottom in enumerate([top,bottom]):
        event = top_bottom.event.item()
        lab=f"{lab_typ[i]} extreme boosted simulation"
        drought = exa.find_start_end_boost(nl_boost,event,start,qu_parent)
        plot_solid_dotted(drought,(nl_boost_right_format.sel(event=event).dropna(dim="time",how="all")/1000),colors[i+2],ax1,end,label=lab)
        if drought.max().item()>0:
            plot_solid_dotted((drought-qu_parent/1000).where(drought - qu_parent/1000 > 0,0).cumsum(),top_bottom[1].item(),colors[i+2],ax2,drought.time.values[-1],cum="True")
        else:
            #plot cumulative threshold exceedence after event (dotted line)
            ax2.axhline(0,color=colors[i+2],linewidth=lws[1],linestyle = lst[1])
    
    # === plot boosted range ===
    #plot instantaneous nl
    ax1.fill_between(
        nl_boost_right_format.time.values,
        (nl_boost_right_format/1000).min("event"), 
        (nl_boost_right_format/1000).max("event"),
        alpha=0.3,color=colors[2],linewidth=0,label="Range of boosted simulations"
    )
    #plot cumulative threshold exceedence
    ax2.fill_between(
        cumsum_boosts[0].time.values,
        cumsum_boosts.min("event"), 
        cumsum_boosts.max("event"),
        alpha=0.3,color=colors[2],linewidth=0
    )
    
    # ===========================
    # === Plot 4 tech drivers ===
    # ===========================
    ax = [ax3,ax5]
    
    # === plot parent ===
    # heating demand and solar PV
    for x,tech in enumerate(["heating-demand","PV"]):
        plot_solid_dotted(parent_tech.sel(time=slice(start,end),technology=tech,value_type="full"), parent_tech.sel(technology=tech,value_type="full"), colors[0], ax[x], end)
    # wind (on and offshore)
    plot_solid_dotted(
        get_on_offshore_wind(parent_tech.sel(value_type="full")).sel(time=slice(start,end)), 
        get_on_offshore_wind(parent_tech.sel(value_type="full")), 
        colors[0], ax4, end
    )
    # storage
    plot_solid_dotted(parent_storage.sel(time=slice(start,end),value_type="full"), parent_storage.sel(value_type="full"), colors[0], ax6, end)
    
    # === Plot boosted top and bottom ===
    for i,top_bottom in enumerate([top,bottom]):
        event = top_bottom.event.item()
        nl_start_end = exa.find_start_end_boost(nl_boost,event,start,qu_parent)
        for j,nl_to_plot in enumerate([nl_start_end, nl_boost]):
            if len(nl_to_plot.where(nl_to_plot>0,drop=True)) == 0: # only plot if there is datat
                continue
            boost_to_plot = tech_boost.stack(event=("lead_time","member")).sel(event=event,time=slice(nl_to_plot.time[0]-timedelta(days=2),nl_to_plot.time[-1]))/1000
            for x,tech in enumerate(["heating-demand","PV"]): 
                boost_to_plot.sel(technology =tech).plot(hue="member",color=colors[i+2],ax=ax[x],linewidth=lws[j],linestyle=lst[j])
            get_on_offshore_wind(boost_to_plot).plot(hue="member",color=colors[i+2],ax=ax4,linewidth=lws[j],linestyle=lst[j])
            (storage_boost.stack(event=("lead_time","member")).sel(event=event,time=slice(nl_to_plot.time[0]-timedelta(days=2),nl_to_plot.time[-1]))/1000).plot(color=colors[i+2],ax=ax6,linewidth=lws[j],linestyle=lst[j])
    
    # === Plot mean +- std dev
    tit = ["Heating demand", "Solar PV"]
    for x,tech in enumerate(["heating-demand","PV"]): 
        plot_mn_std(parent_tech.sel(technology=tech),ax[x],tit[x])
    plot_mn_std(get_on_offshore_wind(parent_tech),ax4,"On- and offshore wind")
    plot_mn_std(parent_storage,ax6,"Hydropower storage")
    
    # ===================
    # === fig configs ===
    # ===================
    plt.title("")
    # xticks
    ticks = [start + timedelta(days=7*i) for i in range(7)]
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([7*i for i in range(7)])
    plt.xlim(start-timedelta(days=1),end+timedelta(days=xlim))
    for ax in [ax1, ax3, ax4, ax5]:
        ax.tick_params(axis="x", labelbottom=False)
    ax1.set_title("Instantaneous net load")
    ax2.set_title("Cumulative net load")
    ax6.set_title("Hydropower storage")
    for i,a in enumerate([ax1,ax2,ax3,ax4,ax5,ax6]):
        pco.set_grid(a)
        a.set_xlabel("")
        if i < 5 and i>1:
            a.set_ylim(ylim)
        if i > 1:
            y=0.89
        else:
            y=0.95
        a.text(0.01,y,string.ascii_lowercase[i],weight="bold",transform=a.transAxes)
    ax1.set_ylabel("Net load [TW]")
    ax2.set_ylabel("Cumulative threshold"+ "\n"+"exceedance [TWh]")
    ax3.set_ylabel("Demand [TW]")
    ax4.set_ylabel("Generation [TW]")
    ax5.set_ylabel("Generation [TW]")
    ax6.set_ylabel("Storage [TWh]")
    ax2.set_xlabel("Time [days since onset]")
    ax6.set_xlabel("Time [days since onset]")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.17)
    hands=[]
    labs = []
    for a in [ax1,ax3]:
        handles, labels = a.get_legend_handles_labels()
        hands.extend(handles)
        labs.extend(labels)
    fig.legend(hands, labs, loc='lower center',ncol=2, frameon=False)
    fig.savefig(f"../../figs_storyline_extremes/boosting_overview_{save_info}.png",bbox_inches="tight",dpi=600,transparent=True)

    
def plot_distrib_with_boosting(ax, cum_scenario,dur_cum_boost,cum_parent,heat,capac):
    """code for plotting figure 5"""        
    #plot distributions
    sns.kdeplot(cum_scenario.values, bw_adjust=0.75,color="k",linewidth=0.7,label="Parent distribution", cut=0,fill=True,ax=ax)
    sns.kdeplot( dur_cum_boost.stack(event=("lead_time","member")).sel(typ="cum").values, bw_adjust=0.75,color="sienna",alpha=0.3,linewidth=0.7,label="Boosted distribution", cut=0,linestyle="dashed",ax=ax,fill=True)
    ax.axvline(cum_parent,linestyle="-",color="k",label="Parent event")
    return ax
    
def plot_temp_z500_maps(times,top,bottom,atm_mn,start,save_info,fsize=(8, 8)):
    #open maps
    ds_temp = open_atm_var_snapshots("temperature",times,top,bottom)
    ds_z500 = open_atm_var_snapshots("Z500",times,top,bottom)
    # plot temperature anomaly maps
    f=(ds_temp.groupby("time.dayofyear")-atm_mn.temperature
      ).plot(
        x="lon",y="lat",row="time",col="event",
        subplot_kws={"projection": ccrs.PlateCarree()},
        transform=ccrs.PlateCarree(),
        add_colorbar=False,
        figsize=fsize,
      )
    # plot z500 maps as contours
    levels = [ [round(int(ds_z500.min()),-1) +x*150 for x in range(int((ds_z500.max()-ds_z500.min())/10)+1)], [round(int(ds_z500.min()),-1) +x*50 for x in range(int((ds_z500.max()-ds_z500.min())/10)+1)]]
    for j, event in enumerate(ds_z500.event):
        for i, time in enumerate(ds_z500.time):
            for n, level in enumerate(levels):
                z = ds_z500.sel(event=event, time=time)
                ax=f.axs[i, j]
                cs = ax.contour(
                    z.lon,z.lat,z,
                    colors="k",
                    levels=level,
                    linewidths = 0.7 - 0.4*n,
                    alpha=1- 0.6*n,
                )
                if n ==0:
                    ax.clabel(cs,inline=True,fontsize=10,levels=cs.levels[::2], inline_spacing=3)
    
    # === fig configs ===
    # add borders and coastlines
    for i,a in enumerate(f.axs.flat):
        if fsize[-1] == 8 and i > 8:
            txt_i = 8
        else:
            txt_i = i
        a = pco.add_country_borders(a)
        a.coastlines(linewidth=0.1)
        a.text(0.01,0.9,string.ascii_lowercase[txt_i],weight="bold",transform=a.transAxes)
    # make it have a tight layout (more difficult with cartopy)
    f.fig.subplots_adjust(left=0.08,right=0.88,bottom=0.08,top=0.92,wspace=0.05,hspace=0.02,)    
    cax = f.fig.add_axes([0.90, 0.08, 0.035, 0.84])
    f.fig.colorbar(
        f.axs[0, 0].collections[0],
        cax=cax,
        label="Temperature anomaly[$^\circ$C]",
    )
    f.set_titles("")
    # add labels to differentiate simulations and dates
    for i, label in enumerate(list(ds_z500.event.values)):
        f.axs[0,i].set_title(label)
    for j, label in enumerate([(d - floor_to_day(start)).days for d in ds_z500.time.values]):
        ax = f.axs[j,0]
        ax.text(
            -0.05, 0.5,label,
            transform=ax.transAxes,
            ha="right",va="center",fontsize=11,rotation=90,
        )
    f.fig.supylabel("Time [days since onset]")
    f.fig.savefig(f"../../figs_storyline_extremes/map_comparison_{save_info}.png",bbox_inches="tight",dpi=600,transparent=True)



# ============================
# === Opening parent files ===
# ============================

print("opening parent files")
path = "/net/xenon/climphys/lbloin/energy_boost/"
nl, nl_qu,extremes, tech_nl,region_nl,storage_nl = exa.open_all_parent_nl(path)
atm,atm_mn = exa.open_atm_vars(path)
print("parent files opened")

# ==============================
# === Boosted case 1: A:2082 ===
# ==============================

boost_dates = ["2081-12-26", "2081-12-29", "2082-01-01", "2082-01-04"]
scenario = "SSP370"
member="A"
start_parent = "2081-12-01"
heat = "fully_electrified"
capac = "future"
climate = "SSP370"

print("opening boosted files")
# get data
nl_boost,tech_boost, storage_boost, qu_parent, nl_parent_only_event,cum_parent, start, end, dur_cum_boost, top,bottom,cum_scenario = open_all_necessary_files_for_boosting(path,boost_dates,start_parent,scenario,member,heat,capac)
parent_tech = tech_nl.sel(heating_scenario=heat,capacity_scenario=capac,climate=scenario,member=member)/1000 #in TW
parent_storage = storage_nl.sel(heating_scenario=heat,capacity_scenario=capac,climate=scenario,member=member).sel(time=slice(start_parent,ut.str_to_cftime_noleap(start_parent) + timedelta(days=80)))/1000
print("boosted files opened")

print("plotting figure 2")
# === Figure 2 ===
plot_boosting_overview(nl_boost,nl_parent_only_event,start,end,qu_parent,cum_parent,top,bottom,parent_tech,parent_storage,tech_boost,storage_boost,f"{capac}_{heat}")
print("figure saved")

print("plotting figure 3 + appendix figure 4")
times = [
    floor_to_day(top.time.item())-timedelta(days=int(top[0])), # start of energy drought
    floor_to_day(bottom.time.item()), # end of best boosted event
    floor_to_day(end), # end of parent event
    floor_to_day(top.time.item()), # end of worst boosted event
]

times_full = [
    floor_to_day(top.time.item())-timedelta(days=int(top[0])), # start of energy drought
    floor_to_day(bottom.time.item()), # end of best boosted event
    floor_to_day(bottom.time.item()) + timedelta(days=7), # end of best boosted event + 1 week,
    floor_to_day(end), # end of parent event
    floor_to_day(end) + timedelta(days=7), # end of parent event + 1 week
    floor_to_day(end) + timedelta(days=14), # end of parent event + 2 weeks
    floor_to_day(top.time.item()), # end of worst boosted event
]
# === Figure 3 ===
plot_temp_z500_maps(times,top,bottom,atm_mn,start,f"{heat}_{capac}")
# === Appendix Figure 5 ===
plot_temp_z500_maps(times_full,top,bottom,atm_mn,start,f"{heat}_{capac}_long",fsize=(8, 12))
print("figures saved")

print("getting data for figure 5a")
# === Figure 5a ===
boost_A_data = [cum_scenario,dur_cum_boost,cum_parent,heat,capac]
print("data retrieved")

# ==============================
# === Boosted case 2: A:2088 ===
# ==============================

boost_dates = ["2088-12-02", "2088-12-05", "2088-12-08", "2088-12-11"]
scenario = "SSP370"
member="A"
start_parent = "2088-11-10"
heat = "current_electrified"
capac = "future_wind_x2"
climate = "SSP370"

print("opening boosted files")
# get data
nl_boost,tech_boost, storage_boost, qu_parent, nl_parent_only_event,cum_parent, start, end, dur_cum_boost, top,bottom,cum_scenario = open_all_necessary_files_for_boosting(path,boost_dates,start_parent,scenario,member,heat,capac)
parent_tech = tech_nl.sel(heating_scenario=heat,capacity_scenario=capac,climate=scenario,member=member)/1000 #in TW
parent_storage = storage_nl.sel(heating_scenario=heat,capacity_scenario=capac,climate=scenario,member=member).sel(time=slice(start_parent,ut.str_to_cftime_noleap(start_parent) + timedelta(days=80)))/1000
print("boosted files opened")

print("plotting figure 4")
# === Appendix figure 4 ===
plot_boosting_overview(nl_boost,nl_parent_only_event,start,end,qu_parent,cum_parent,top,bottom,parent_tech,parent_storage,tech_boost,storage_boost,f"{capac}_{heat}",ylim=[0,1.9],xlim=12)
print("figure saved")

print("plotting appendix figure 5")
times = [
    floor_to_day(top.time.item())-timedelta(days=int(top[0])), # start of energy drought
    floor_to_day(top.time.item()), # end of worst boosted event
    floor_to_day(end), # end of parent event
]

# === Appendix figure 5 ===
plot_temp_z500_maps(times,top,bottom,atm_mn,start,f"{heat}_{capac}",fsize=(8, 6))
print("figure saved")

print("getting data for figure 5b")
# === Figure 5b ===
boost_B_data = [cum_scenario,dur_cum_boost,cum_parent,heat,capac]
print("data retrieved")

print("plotting figure 5")
f,axs=plt.subplots(1,2,figsize=(8,3.2))
plot_distrib_with_boosting(axs[0],*boost_A_data)
plot_distrib_with_boosting(axs[1],*boost_B_data)

#fig configs
axs[0].set_ylabel("Probability density")
axs[1].set_ylabel("")
for i,a in enumerate(axs):
    a.set_xlabel("Cumulative threshold exceedence [TWh]")
    a.set_title("")
    pco.set_grid(a)
    a.text(0.015,0.94,string.ascii_lowercase[i],weight="bold",transform=a.transAxes)
    a.set_xlim(0,None)
plt.tight_layout()
plt.subplots_adjust(bottom=0.25)
handles, labels = axs[0].get_legend_handles_labels()
f.legend(handles, labels, loc='lower center',ncol=3, frameon=False)
f.savefig(f"../../figs_storyline_extremes/distribution_with_boosting_compare.png",bbox_inches="tight",dpi=600,transparent=True)
print("figure saved")

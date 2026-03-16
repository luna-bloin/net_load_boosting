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
from datetime import timedelta
import string
import matplotlib.dates as mdates

# =================
# === Functions ===
# =================

def open_all_necessary_files_for_boosting(path,boost_dates,start_parent,scenario,member,heat,capac):
    """Opens all necessary data for doing the three types of boosting plot"""
    # open boosted files 
    nl_boost = exa.open_boost(path,boost_dates,start_parent,scenario,member,typ="transmission").sel(heating_scenario=heat,capacity_scenario=capac)
    tech_boost = exa.open_boost(path,boost_dates,start_parent,scenario,member,typ="eng_vars").sel(heating_scenario=heat,capacity_scenario=capac)
    # find net load in correct scenario
    nl_parent = nl.sel(heating_scenario=heat,capacity_scenario=capac,climate=climate)
    qu_parent = nl_qu.sel(heating_scenario=heat,capacity_scenario=capac)
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
    top_5 = ds.sortby(ds.sel(typ="cum"),ascending=False).isel(event=slice(0,5))
    bottom_5 = ds.sortby(ds.sel(typ="cum"),ascending=False).isel(event=slice(-5,None))
    return nl_boost,tech_boost, qu_parent, nl_parent_only_event,cum_parent, start, end, dur_cum_boost, top_5,bottom_5

def plot_main_mn_std(to_plot,ax,title):
    to_plot.sel(value_type="full").plot(color="k",ax=ax)
    to_plot.sel(value_type="mean").plot(color="k",ax=ax)
    ax.fill_between(to_plot.time.values,to_plot.sel(value_type="mean") - to_plot.sel(value_type="std"),to_plot.sel(value_type="mean") + to_plot.sel(value_type="std"),color="k",alpha=0.3)
    ax.set_title(title)

def get_on_offshore_wind(ds):
    return ds.sel(technology=["Wind_onshore","Wind_offshore"]).sum("technology", skipna=False)

def plot_net_load_boosted_overview(nl_boost,nl_parent_only_event,top_5,bottom_5,start,end,qu_parent,heat,capac,xlim_end,xticks):
    """code for plotting figure 4a and appendix figure 9a"""
    f,ax=plt.subplots(figsize=(8,2.8))
    #plot range of boosted simulations
    to_plot = nl_boost.stack(event=("lead_time","member")).dropna(dim="time",how="all").sel(time=slice(None,end+timedelta(days=23))).convert_calendar("proleptic_gregorian")/1000
    plt.fill_between(to_plot.time.values,to_plot.min("event"), to_plot.max("event"),alpha=0.5,color=pco.colors[0],linewidth=0)
    # plot parent net load
    (nl_parent_only_event.convert_calendar("proleptic_gregorian")/1000).plot(color="k",linewidth=1,label="Parent event",zorder=4)
    # plot top and bottom 5 events of boosted net load
    colors = ["sienna","olive"]
    lab_typ = ["Most", "Least"]
    for i,top_bottom in enumerate([top_5,bottom_5]):
        for j,event in enumerate(top_bottom.event.values):
            if j*4 ==0:
                lw = 0.8
                lab=f"{lab_typ[i]} extreme simulations"
            else:
                lw=0.3
                lab="_no"
            exa.find_start_end_boost(nl_boost,event,start,qu_parent).convert_calendar("proleptic_gregorian").plot(linewidth=lw,color=colors[i],label=lab)
    # plot threshold and parent event end
    plt.axhline(qu_parent/1000,linestyle="--",color="k",linewidth=0.7)
    ax.axvline(end,linestyle="-",color="k",linewidth=0.7)
    #fig configs
    plt.xlim(start-timedelta(days=2),end+timedelta(days=xlim_end))
    plt.title("")
    plt.ylabel("Net load [TW]")
    plt.xlabel("Time")
    pco.set_grid(ax)
    ax.legend()
    ax.text(0.01,0.93,string.ascii_lowercase[0],weight="bold",transform=ax.transAxes)
    ticks = to_plot.time.sel(time=xticks).values
    ax.set_xticks(ticks)
    ax.set_xticklabels(xticks)
    f.tight_layout()
    f.savefig(f"../../figs_storyline_extremes/overview_boosting_with_top_bottom_{heat}_{capac}.svg",bbox_inches="tight",dpi=600,transparent=True)

def plot_tech_breakdown_boosting(parent_tech,tech_boost,top_5,bottom_5,nl_boost,start,end,qu_parent,heat,capac,xlim_end,xticks):
    """code for plotting figure 4b-c and appendix figure 9b-c"""
    f,ax=plt.subplots(2,2,figsize=(8,3.2),sharex=True,sharey=True)
    lw = 0.3
    # === plot top and bottom 5 boosted ===
    colors = ["sienna","olive"]
    for i,top_bottom in enumerate([top_5,bottom_5]):
        for event in top_bottom.event.values:
            nl_start_end = exa.find_start_end_boost(nl_boost,event,start,qu_parent).dropna(dim= "time",how="all")
            if len(nl_start_end.time) > 0:
                boost_to_plot = tech_boost.stack(event=("lead_time","member")).sel(event=event,time=slice(nl_start_end.time[0]-timedelta(days=2),nl_start_end.time[-1]))/1000 #in TW
                boost_to_plot.sel(technology ="heating-demand").plot(color=colors[i],ax=ax[i][0],linewidth=lw)
                get_on_offshore_wind(boost_to_plot).plot(color=colors[i],ax=ax[i][1],linewidth=lw)
    
    # === plot parent ===
    for i in range(2):
        plot_main_mn_std(parent_tech.sel(technology ="heating-demand"),ax[i][0],"Heating demand") #heating demand
        plot_main_mn_std(get_on_offshore_wind(parent_tech),ax[i][1],"On- and offshore wind") # on+offshore wind
    
    #fig configs
    for i,a in enumerate(ax.flatten()):
        a.set_xlim(start-timedelta(days=2),end+timedelta(days=22))
        a.axvline(end,linestyle="-",color="k",linewidth=0.7,zorder=4)
        if i %2 == 0:
            a.set_ylabel("Demand [TW]")
        else:
            a.set_ylabel("Generation [TW]")
        if i >1:
            a.set_title("")
            a.set_xlabel("Time")
        else:
            a.set_xlabel("")
        a.text(0.01,0.87,string.ascii_lowercase[i+1],weight="bold",transform=a.transAxes)
        plt.xlim(start-timedelta(days=2),end+timedelta(days=xlim_end))
        ticks = parent_tech.time.sel(time=xticks).values
        ticks = [ticks[0],ticks[-1]]
        a.set_xticks(ticks)
    f.tight_layout()
    f.savefig(f"../../figs_storyline_extremes/tech_breakdown_boosting_with_top_bottom_{heat}_{capac}.png",bbox_inches="tight",dpi=600,transparent=True)

def plot_distrib_with_boosting(extremes,dur_cum_boost,cum_parent,heat,capac,string_type,xlim_end):
    """code for plotting figure 5"""
    f,ax=plt.subplots(figsize=(7,3.2))

    # find shortfall events for historical period
    cum_hist=extremes.sel(heating_scenario=heat,capacity_scenario=capac,climate="historical").dropna(dim="event",how="all").sel(typ="cum")
    cum_ssp=extremes.sel(heating_scenario=heat,capacity_scenario=capac,climate="SSP370").dropna(dim="event",how="all").sel(typ="cum")
    
    #plot parent distributions
    sns.kdeplot(cum_hist, bw_adjust=1,color="k",linewidth=0,label="Historical climate",fill=True, cut=0)
    sns.kdeplot(cum_ssp.values, bw_adjust=1,color=pco.colors[0],linewidth=0,label="End-of-century climate", cut=0,fill=True)
    # plot boosted distribution
    sns.kdeplot(dur_cum_boost.stack(event=("lead_time","member")).sel(typ="cum").values, bw_adjust=1,color=pco.colors[0],linewidth=1,label="Boosted simulations", cut=0)
    #plot parent event
    plt.axvline(cum_parent,linestyle="--",color=pco.colors[0],label="Parent event")
    
    #fig configs
    plt.ylabel("Probability density")
    plt.xlabel("Cumulative threshold exceedence [TWh]")
    plt.title("")
    f.legend(ncols=2,loc="lower center")
    pco.set_grid(ax)
    plt.xlim(0,xlim_end)
    f.tight_layout()
    plt.subplots_adjust(bottom=0.32)
    ax.text(0.01,0.92,string.ascii_lowercase[string_type],weight="bold",transform=ax.transAxes)
    f.savefig(f"../../figs_storyline_extremes/distribution_with_boosting_{heat}_{capac}.svg",bbox_inches="tight",dpi=600,transparent=True)


# ============================
# === Opening parent files ===
# ============================

print("opening parent files")
path = "/net/xenon/climphys/lbloin/energy_boost/"
nl, nl_qu,extremes, tech_nl,region_nl,storage_nl = exa.open_all_parent_nl(path)
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
nl_boost,tech_boost, qu_parent, nl_parent_only_event,cum_parent, start, end, dur_cum_boost, top_5,bottom_5 = open_all_necessary_files_for_boosting(path,boost_dates,start_parent,scenario,member,heat,capac)
parent_tech = tech_nl.sel(heating_scenario=heat,capacity_scenario=capac,climate=scenario,member=member)/1000 #in TW
print("boosted files opened")

print("plotting figure 4 + 5a")
# === Figure 4a ===
plot_net_load_boosted_overview(nl_boost,nl_parent_only_event,top_5,bottom_5,start,end,qu_parent,heat,capac,22,["2082-01-11", "2082-01-31"])
# === Figure 4b-c ===
plot_tech_breakdown_boosting(parent_tech,tech_boost,top_5,bottom_5,nl_boost,start,end,qu_parent,heat,capac,22,slice("2082-01-11", "2082-01-31"))
# === Figure 5a ===
plot_distrib_with_boosting(extremes,dur_cum_boost,cum_parent,heat,capac,0,250)
print("figures saved")

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
nl_boost,tech_boost, qu_parent, nl_parent_only_event,cum_parent, start, end, dur_cum_boost, top_5,bottom_5 = open_all_necessary_files_for_boosting(path,boost_dates,start_parent,scenario,member,heat,capac)
parent_tech = tech_nl.sel(heating_scenario=heat,capacity_scenario=capac,climate=scenario,member=member)/1000 #in TW
print("boosted files opened")

print("plotting figure 5b + appendix figure 9")
# === Appendix Figure 9a ===
plot_net_load_boosted_overview(nl_boost,nl_parent_only_event,top_5,bottom_5,start,end,qu_parent,heat,capac,1,["2088-12-20", "2088-12-26"])
# === Appendix Figure 9b-c ===
plot_tech_breakdown_boosting(parent_tech,tech_boost,top_5,bottom_5,nl_boost,start,end,qu_parent,heat,capac,1,slice("2088-12-20", "2088-12-26"))
# === Figure 5b ===
plot_distrib_with_boosting(extremes,dur_cum_boost,cum_parent,heat,capac,1,10)
print("figures saved")
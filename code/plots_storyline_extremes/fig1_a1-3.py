import sys
sys.path.append("../utils/")
# local imports
import utils as ut
import plot_config as pco
import extreme_analysis as exa
#standard
import numpy as np
import pandas as pd
#plotting
import matplotlib.pyplot as plt
import seaborn as sns
#other 
from tqdm import tqdm
import string

# =================
# === Functions ===
# =================

def find_max_nl_for_violins(extremes,nl,heat,capac):
    """get the maximum net load (as a pd dataframe, necessary for sns violinplot) for each energy shortfall event in the dataset extremes, for a given heating scenario heat and capacity scenario capac"""
    extremes_scen = extremes.sel(heating_scenario=heat,capacity_scenario=capac)
    nl_scen = nl.sel(heating_scenario=heat,capacity_scenario=capac)/1000
    peak_nls = []
    for climate in extremes_scen.climate:
        extremes_here = extremes_scen.sel(climate=climate).dropna(dim="event", how="all")
        nl_here = nl_scen.sel(climate=climate)
        for event in tqdm(extremes_here.event):
            end = extremes_here.sel(typ="cum",event=event).time
            duration = extremes_here.sel(typ="dur",event=event)
            mem = extremes_here.member.sel(event=event).item()
            peak = exa.find_peak(nl_here,mem,end,duration)
            peak_nls.append({
                    "climate": climate.item(),
                    "Maximum net load [TW]": peak
                })
    return pd.DataFrame(peak_nls)

def find_dur_doy_for_violins(extremes,heat,capac):
    """"get the duration and day of year of the end (as a pd dataframe, necessary for sns violinplot) for each energy shortfall event in the dataset extremes, for a given heating scenario heat and capacity scenario capac"""
    extreme_scen = extremes.sel(heating_scenario=heat,capacity_scenario=capac)
    dur_plot = extreme_scen.sel(typ="dur").dropna(dim="event",how="all").to_series().reset_index(name="Duration [days]")
    cum_only = extreme_scen.sel(typ="cum").dropna(dim="event",how="all")
    doy_plot = cum_only.time.dt.dayofyear.broadcast_like(cum_only).where(cum_only.notnull()).to_series().reset_index(name="Day of year")
    return dur_plot,doy_plot

def plot_violin_clim_halves(to_plot,ax,x):
    sns.violinplot(
                data=to_plot,
                x=x,
                y=to_plot.columns[-1],
                hue="climate",
                split=True,
                cut=0,
                inner="quart",
                ax=ax,
                palette = [pco.colors[1],pco.colors[0]],
                alpha=0.5,
            )
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel("")       
    ax.legend_.remove()
    pco.set_grid(ax) 
    return None
    
# ===============================
# === Opening the right files ===
# ===============================

print("opening files")
path = "/net/xenon/climphys/lbloin/energy_boost/"
nl, nl_qu,extremes = exa.open_all_parent_nl(path,nl_only=True)
capacity_scenarios = nl.capacity_scenario.values
heating_scenarios = nl.heating_scenario.values
print("files opened")

# ================
# === Figure 1 ===
# ================
print("plotting figure 1")
fig = plt.figure(figsize=(7.2,4.5))

two_representative_scenarios = [["future","fully_electrified"],["future_wind_x2","current_electrified"]]
for i,scen in enumerate(two_representative_scenarios):
    heat = scen[1]
    capac = scen[0]
    # set up figure
    ax_int = fig.add_subplot(2, 3, 1 + 3*i)
    ax_dur = fig.add_subplot(2, 3, 2 + 3*i)
    ax_doy = fig.add_subplot(2, 3, 3 + 3*i, projection="polar")

    # get data for plots
    dur_plot,doy_plot = find_dur_doy_for_violins(extremes,heat,capac)
    max_plot = find_max_nl_for_violins(extremes,nl,heat,capac)
    
    #plot maximum net load and duration violin plots for each event
    plot_violin_clim_halves(max_plot,ax_int,0)
    plot_violin_clim_halves(dur_plot,ax_dur,0)
    ax_int.set_xticks([])
    ax_dur.set_xticks([])
    
    # get seasonality data and plot it
    ymax=0
    for x,climate in enumerate(['historical','SSP370']):
        # get current set of top events
        nl_one_scen = nl.sel(heating_scenario=heat,capacity_scenario=capac,climate=climate)/1000
        qu = nl_qu.sel(capacity_scenario=capac,heating_scenario=heat)
        extreme_here = extremes.sel(heating_scenario=heat,capacity_scenario=capac,climate=climate).dropna(dim="event",how="all")
        dur = extreme_here.sel(typ = "dur")
        cum = extreme_here.sel(typ="cum")
        cum = cum.sortby(cum,ascending=False)
        doy = cum.time.dt.dayofyear.values
        theta = 2 * np.pi * (doy - 1) / 365
        for cum_ext in cum[0:5]:
            #plot top five events in terms of maximum net load and duration
            nl_ext = exa.find_peak(nl_one_scen,cum_ext.member,cum_ext.time,dur.sel(time=cum_ext.time))
            ax_int.plot((-1+x*2)*0.005,nl_ext,"o",markersize=3.5,zorder=4,color=pco.colors[1-x],mec="k",mew=0.7)
            ax_dur.plot((-1+x*2)*0.005,dur.sel(time=cum_ext.time),"o",markersize=3.5,zorder=4,color=pco.colors[1-x],mec="k",mew=0.7)
        # plotting seasonality
        vals=ax_doy.hist(theta, bins=48,density=True,zorder=4,color=pco.colors[1-x],alpha=0.5,label=ut.scen_config_dict[climate])  # 24 bins ≈ half-month resolution
        for th in theta[0:5]:
            ax_doy.plot(th,vals[0].max(),"o",markersize=3.5,zorder=4,color=pco.colors[1-x],mec="k",mew=0.7)
        ymax = max(ymax,vals[0].max())
    
    # Fig config 
    ax_doy.set_theta_zero_location("N")   # Jan at top
    ax_doy.set_theta_direction(-1)        # clockwise (calendar style)
    # Day of year at start of each month
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_mids   = [16, 46, 75, 106, 136, 167, 197, 228, 259, 289, 320, 350]
    month_letters = ['J','F','M','A','M','J','J','A','S','O','N','D']
    # Convert to radians
    ticks_rad = [d / 365 * 2 * np.pi for d in month_starts]
    mids_rad  = [d / 365 * 2 * np.pi for d in month_mids]
    # Ticks at month starts, labels at month midpoints
    ax_doy.set_xticks(ticks_rad)
    ax_doy.set_xticklabels([])  # no labels on ticks
    # Add month letter labels at midpoints
    for angle, letter in zip(mids_rad, month_letters):
        ax_doy.text(angle, ymax*1.2, letter,
                ha='center', va='center', fontsize=10)
    ax_doy.set_yticklabels([])
    pco.set_grid(ax_doy)
    if i ==1:
        handles, labels = ax_doy.get_legend_handles_labels() 
        fig.legend(handles, labels, loc='lower center',ncol=5,frameon=False)
axes = fig.get_axes()
axes[3].sharey(axes[0])
axes[4].sharey(axes[1])
for i,a in enumerate(axes):
    a.text(0.01,0.95,string.ascii_lowercase[i],weight="bold",transform=a.transAxes)

fig.tight_layout()
fig.subplots_adjust(bottom=0.1)
fig.savefig(f"../../figs_storyline_extremes/extreme_charac_2_scen.svg",bbox_inches="tight",dpi=600,transparent=True)

print("figure 1 saved")
# ============================
# === Appendix figures 1-3 ===
# ============================

print("plotting appendix figures 1-3")
f_int,ax_int = plt.subplots(2,4,figsize=(7.2,4),sharey=True) #plot intensity (maximum net load) for all scenarios 
f_dur,ax_dur = plt.subplots(2,4,figsize=(7.2,4),sharey=True) # plot persistence (duration) for all scenarios
f_doy,ax_doy = plt.subplots(2,4,figsize=(7.2,4),subplot_kw={"projection":"polar"}) # plot seasonality (day of year of event end) for all scenarios

# iterate over all scenarios
for i,heat in enumerate(heating_scenarios):
    for j,capac in enumerate(capacity_scenarios):
        print(heat, capac)
        
        # get data for intensity and duration plots
        dur_plot,doy_plot = find_dur_doy_for_violins(extremes,heat,capac)
        max_plot = find_max_nl_for_violins(extremes,nl,heat,capac)
        
        #plot maximum net load and duration violin plots for each scenario
        plot_violin_clim_halves(max_plot,ax_int[i][j],0)
        plot_violin_clim_halves(dur_plot,ax_dur[i][j],0)
    
        # get seasonality data and plot it (needs to be done for each climate separately)
        ymax=0
        for x,climate in enumerate(['historical','SSP370']):
            # get current set of energy shortfall events
            nl_one_scen = nl.sel(heating_scenario=heat,capacity_scenario=capac,climate=climate)/1000
            qu = nl_qu.sel(capacity_scenario=capac,heating_scenario=heat)
            extreme_here = extremes.sel(heating_scenario=heat,capacity_scenario=capac,climate=climate).dropna(dim="event",how="all")
            # find duration, intensity and dayofyear of each event
            dur = extreme_here.sel(typ = "dur")
            cum = extreme_here.sel(typ="cum")
            cum = cum.sortby(cum,ascending=False)
            doy = cum.time.dt.dayofyear.values
            theta = 2 * np.pi * (doy - 1) / 365
            # plot the top 5 events in terms of intensity and duration
            for cum_ext in cum[0:5]:
                nl_ext = exa.find_peak(nl_one_scen,cum_ext.member,cum_ext.time,dur.sel(time=cum_ext.time))
                ax_int[i][j].plot((-1+x*2)*0.005,nl_ext,"o",markersize=3.5,zorder=4,color=pco.colors[1-x],mec="k",mew=0.7)
                ax_dur[i][j].plot((-1+x*2)*0.005,dur.sel(time=cum_ext.time),"o",markersize=3.5,zorder=4,color=pco.colors[1-x],mec="k",mew=0.7)
            
            # plot day of year rose plot
            vals=ax_doy[i][j].hist(theta, bins=48,density=True,zorder=4,color=pco.colors[1-x],alpha=0.5,label=ut.scen_config_dict[climate])  # 48 bins ≈ quarter-month resolution
            for th in theta[0:5]: #plot top 5 events in rose plot
                ax_doy[i][j].plot(th,vals[0].max(),"o",markersize=3.5,zorder=4,color=pco.colors[1-x],mec="k",mew=0.7)
            ymax = max(ymax,vals[0].max())
        #fig configs
        ax_doy[i][j].set_theta_zero_location("N")   # Jan at top
        ax_doy[i][j].set_theta_direction(-1)        # clockwise (calendar style)
        # Day of year at start of each month
        month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        month_mids   = [16, 46, 75, 106, 136, 167, 197, 228, 259, 289, 320, 350]
        month_letters = ['J','F','M','A','M','J','J','A','S','O','N','D']
        # Convert to radians
        ticks_rad = [d / 365 * 2 * np.pi for d in month_starts]
        mids_rad  = [d / 365 * 2 * np.pi for d in month_mids]
        # Ticks at month starts, labels at month midpoints
        ax_doy[i][j].set_xticks(ticks_rad)
        ax_doy[i][j].set_xticklabels([])  # no labels on ticks
        # Add month letter labels at midpoints
        for angle, letter in zip(mids_rad, month_letters):
            ax_doy[i][j].text(angle, ymax*1.2, letter,
                    ha='center', va='center', fontsize=10)
        ax_doy[i][j].set_yticklabels([])
        pco.set_grid(ax_doy[i][j])

#fig config
for i, a in enumerate([ax_int,ax_dur,ax_doy]):
    for n,ax in enumerate(a.flatten()):
        if i < 2: #only for intensity and duration plot
            ax.set_xticks([])
            ax.set_ylabel("")
        ax.text(0.01,0.97,string.ascii_lowercase[n],weight="bold",transform=ax.transAxes)

lab = ["Maximum net load [TW]", "Duration [days]"]
for i, fig in enumerate([f_int,f_dur]):
    fig.text(0.01,0.5,lab[i],rotation='vertical',verticalalignment='center', horizontalalignment='center')

typ = ["intensity","persistence","seasonality"]
for i,fig in enumerate([f_int,f_dur,f_doy]):
    handles, labels = ax_doy[0][0].get_legend_handles_labels() 
    fig.legend(handles, labels, loc='lower center',ncol=5,frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.1)
    fig.savefig(f"../../figs_storyline_extremes/{typ[i]}_cc.svg",bbox_inches="tight",dpi=600,transparent=True)

print("appendix figures 1-3 saved")

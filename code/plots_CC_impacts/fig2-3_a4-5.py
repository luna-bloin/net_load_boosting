import sys
sys.path.append("../utils/")
import utils as ut
import plot_config as pco
import xarray as xr
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
from matplotlib.gridspec import GridSpec
import numpy as np
import string

# =================
# === Functions === 
# =================

def plot_p90(typ,ax,p90,scenario,i):
    """
    Plots all p90 events either as day of year or hour of day (specified through typ) for a given scenario and for a figure ax and color i
    """
    if typ == "doy":
        doy = p90.groupby("time.dayofyear").mean()
        to_plot = ut.get_smoothed_doy(doy,21) # smooth with a 21 day rolling mean
        ax.fill_between(to_plot.dayofyear,0,to_plot,alpha=0.3,color=pco.colors[1-i],label=ut.scen_config_dict[scenario])

    elif typ == "hod":
        to_plot = p90.groupby("time.hour").mean()
        ax.fill_between(to_plot.hour,0,to_plot,alpha=0.3,color=pco.colors[1-i],label=ut.scen_config_dict[scenario])

    print(to_plot.sum().values)
    ax.set_title(f"")
    if typ == "doy":
        ax.set_xticks([60,152,244,335],labels=["Mar","Jun","Sep","Dec"])
    elif typ == "hod": 
        ax.set_xlabel("Hour of day")
        ax.set_xticks([3,9,15,21])
    #plot configs
    ax.set_ylabel("")

# ==================
# === Open files ===
# ==================

path = "/net/xenon/climphys/lbloin/energy_boost/"
# get net load for different scenarios and assumptions
nl = {}
for scenario in  ut.CESM2_REALIZATION_DICT: 
    nl_scen = {
        "copperplate":xr.open_dataset(f"{path}net_load_hydro_storage_{scenario}.nc").net_load_adjusted,
        "realistic_transmission":xr.open_dataset(f"{path}net_load_transmission_{scenario}.nc").net_load
    }
    nl[scenario] = nl_scen

#list of different scenarios and assumptions
transmission_types = ["copperplate","realistic_transmission"]
capacity_scenarios = nl[scenario]["copperplate"].capacity_scenario.values
heating_scenarios = nl[scenario]["copperplate"].heating_scenario.values
scenario_configs = [["current","current_electrified"],["future","fully_electrified"],["future_wind_x2","current_electrified"]]

# get 90th percentile for all climate model data combined, but separate for each scenario
qu_all = []
for typ in transmission_types:
    qu_capac = []
    for x,capacity_scenario in enumerate(capacity_scenarios):
        qu_heat = []
        for j,heating_scenario in enumerate(heating_scenarios):
            nl_for_qu = []
            for scenario in nl.keys():
                nl_for_qu.append(nl[scenario][typ].sel(capacity_scenario=capacity_scenario,heating_scenario=heating_scenario).dropna(dim='time').values)
            qu_heat.append(np.quantile(np.array(nl_for_qu),0.9))
        qu_capac.append(qu_heat)
    qu_all.append(qu_capac)
qu_all = xr.DataArray(
    qu_all,
    dims=["transmission_type","capacity_scenario","heating_scenario"],
    coords = {"transmission_type":transmission_types,"capacity_scenario":capacity_scenarios,"heating_scenario":heating_scenarios}
)

# =============
# === Fig 2 ===
# =============

#find common bins
nl_vals = []
for scenario in  ut.CESM2_REALIZATION_DICT:
    nl_vals.append(nl[scenario]["copperplate"].stack(dim=("time","member","heating_scenario","capacity_scenario")).values)
bins = np.histogram_bin_edges(nl_vals, bins='auto')

# figure
fig = plt.figure(figsize=(7.2, 4))
gs = GridSpec(4, 4, figure=fig,height_ratios=[1,0.1,1,0.18])
ylims =  [((0, 5200),(17500,19000)),((0, 5200),(29500,31000))]# 0.0013), (0.004, 0.0043)),((0, 0.0013), (0.007, 0.0073))]
for x, heating_scenario in enumerate(heating_scenarios):
    for j, capacity_scenario in enumerate(capacity_scenarios):
        spec = gs[x*2, j]
        # consistent broken y-limits
        bax = brokenaxes(ylims=ylims[x], hspace=0.08, subplot_spec=spec)

        for i, scenario in enumerate(ut.CESM2_REALIZATION_DICT):
            # Select relevant data
            data = nl[scenario]["copperplate"].sel(
                heating_scenario=heating_scenario,
                capacity_scenario=capacity_scenario
            )
            # Plot histogram on brokenaxes
            bax.hist(
                data.stack(dim=("time","member")).values,
                histtype="step",
                color=pco.colors[1 - i],
                label=f"{ut.scen_config_dict[scenario]}",
                bins=bins,
                linewidth=0.8,
            )

        # add p90 as vertical line
        bax.axvline(
            qu_all.sel(
                heating_scenario=heating_scenario,
                capacity_scenario=capacity_scenario,
                transmission_type="copperplate"
            ),
            linestyle = "dashed",
            color="k",
            linewidth=0.5,
            label = "P\u2089\u2080"
        )

        # Only show y-axis ticks on the leftmost column
        if j != 0:
            for axis in bax.axs:
                axis.tick_params(labelleft=False)
                
        if x ==1:
            bax.axs[1].set_xlabel("Net load [GWh]")
        for i,a in enumerate(bax.axs):
            pco.set_grid(a)
            if i ==0:
                a.text(0.02,0.67,string.ascii_lowercase[x*4+j%4],weight="bold",transform=a.transAxes)
fig.text(0.01,0.5,"Frequency of occurence",rotation='vertical',verticalalignment='center', horizontalalignment='center')
    
handles, labels = bax.axs[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center',ncol=4, frameon=False)
plt.savefig(f"../../figs_CC_impacts/fig2.png",bbox_inches="tight", dpi = 600,transparent=True)

# =============
# === Fig 3 ===
# =============

# figure
f,ax=plt.subplots(2,3,figsize=(7.2,5),sharey=True)
for j,config in enumerate(scenario_configs):
    for i, scenario in enumerate(nl.keys()): #open hist and SSP370
        # get net load
        net_load = nl[scenario]["realistic_transmission"].sel(capacity_scenario=config[0],heating_scenario=config[1])
        net_load_stacked = net_load.stack(event=("member","time")).dropna(dim="event")
        # percentile to work with
        qu_nl = qu_all.sel(capacity_scenario=config[0],heating_scenario=config[1],transmission_type="realistic_transmission")
        # frequency of events above percentile
        ext = xr.where(net_load_stacked>qu_nl,1,0)
        #print info
        print(ext.sum().values)
        print(config, scenario)
        # dayofyear
        plot_p90("doy",ax[0][j],ext,scenario,i)
        #hour of day
        plot_p90("hod",ax[1][j],ext,scenario,i)
    for x in [0,1]:
        pco.set_grid(ax[x][j])
        ax[x][j].text(0.02,0.95,string.ascii_lowercase[j+x*3],weight="bold",transform=ax[x][j].transAxes)
        ax[x][j].spines['right'].set_visible(False)
        ax[x][j].spines['top'].set_visible(False)
f.text(0.01,0.6,"Frequency of events above P\u2089\u2080",rotation='vertical',verticalalignment='center', horizontalalignment='center',fontsize=12)
handles, labels = ax[0][0].get_legend_handles_labels()
f.legend(handles, labels, loc='lower center',ncol=2, frameon=False)
plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.savefig(f"../../figs_CC_impacts/fig3.png",bbox_inches ="tight", dpi = 600)

# =============================
# === Appendix figs 4 and 5 ===
# =============================

# figure
f,ax=plt.subplots(2,4,figsize=(7.2,3),sharex=True,sharey=True)
f2,ax2=plt.subplots(2,4,figsize=(7.2,3),sharex=True,sharey=True)
typ = "realistic_transmission" 
for x, heating_scenario in enumerate(heating_scenarios):
        for j, capacity_scenario in enumerate(capacity_scenarios):
            for i, scenario in enumerate(nl.keys()): #open hist and SSP370
                # get net load
                net_load = nl[scenario][typ].sel(capacity_scenario=capacity_scenario,heating_scenario=heating_scenario)
                net_load_stacked = net_load.stack(event=("member","time")).dropna(dim="event")
                # percentile to work with
                qu_nl = qu_all.sel(capacity_scenario=capacity_scenario,heating_scenario=heating_scenario,transmission_type=typ)
                # frequency of events above percentile
                ext = xr.where(net_load_stacked>qu_nl,1,0)
                #print seasonal info
                print(heating_scenario,capacity_scenario, scenario)
                # dayofyear
                plot_p90("doy",ax[x][j],ext,scenario,i)
                #hour of day
                plot_p90("hod",ax2[x][j],ext,scenario,i)
                if x==0:
                    ax[x][j].set_xlabel("")
                    ax2[x][j].set_xlabel("")
            for a in [ax,ax2]:
                pco.set_grid(a[x][j])
                a[x][j].text(0.02,0.95,string.ascii_lowercase[j+x*4],weight="bold",transform=a[x][j].transAxes)
                a[x][j].spines['right'].set_visible(False)
                a[x][j].spines['top'].set_visible(False)
save_typ = ["seasonal","daily"]
axs = [ax,ax2]
for i,fig in enumerate([f,f2]):
    fig.text(0.01,0.6,"Frequency of events above P\u2089\u2080",rotation='vertical',verticalalignment='center', horizontalalignment='center',fontsize=12)
    handles, labels = axs[i][0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',ncol=2,frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2+0.05*i)
    fig.savefig(f"../../figs_CC_impacts/appendix_fig{4+i}.png",bbox_inches ="tight", dpi = 600)

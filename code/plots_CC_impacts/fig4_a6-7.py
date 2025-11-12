import sys
sys.path.append("../utils/")
import utils as ut
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import plot_config as pco
import string
import seaborn as sns
import pandas as pd
from scipy import stats

# =================
# === Functions ===
# =================

def get_all_nl_above_p90(qu_all, nl,heating_scenario,capacity_scenario,season="all"):
    """
    find all net load events that are above p90 and return the net load as a function of technology as a pandas dataframe
    """
    df_list = []
    for scenario in ut.CESM2_REALIZATION_DICT:
        # find p90
        qu_transm= qu_all.sel(capacity_scenario=capacity_scenario,heating_scenario=heating_scenario)
        # find transmission nl values above p90
        transmission = nl[scenario]["transmission"].sel(
                            heating_scenario=heating_scenario,
                            capacity_scenario=capacity_scenario
                        ).dropna(dim="time",how="all").stack(dim=("time","member"))
        transm_qu90 = transmission.where(transmission>=qu_transm,drop=True)
        # find nl values separated by tech, that match when transmission is above p90
        tech = nl[scenario]["by_tech"].sel(heating_scenario=heating_scenario,capacity_scenario=capacity_scenario).dropna(dim="time",how="all").stack(dim=("time","member")).where(transm_qu90,drop=True)
        if season != "all":
            tech = tech.groupby("time.season")[season]
        tech_anom = tech - nl_tech_mn.sel(heating_scenario=heating_scenario,capacity_scenario=capacity_scenario) # anomaly from mean value
        # Safely convert to tidy dataframe with a "value" column
        df_scen = tech_anom.to_series().reset_index(name="value")
        df_scen["scenario"] = ut.scen_config_dict[scenario]
        df_list.append(df_scen)
    return pd.concat(df_list, ignore_index=True)

def plot_violin(axs,df_all):
    """
    plots violin plots of all events in df_all by technology, separating generation and demand
    """
    for x,tech_type in enumerate([generation,demand]):
        ax=axs[x]
        pco.set_grid(ax) 
        plot_here = df_all[df_all["technology"].isin(tech_type)]
        mean_df = plot_here.groupby(['technology', 'scenario'], as_index=False)['value'].mean()
        sns.violinplot(
            data=plot_here,
            x="technology",
            y="value",
            hue="scenario",
            split=True,
            cut=0,
            density_norm="width",
            inner="quart",
            ax=ax,
            palette = [pco.colors[1],pco.colors[0]],
            alpha=0.5,
            order=tech_type,
        )
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.axhline(0,color="k",linewidth=0.5,linestyle="--")
        ax.set_xlabel("")            
        ax.legend_.remove()

# ==================
# === Open files ===
# ==================

path = "/net/xenon/climphys/lbloin/energy_boost/"
# get net load for all scenarios/climate periods, with different resolution (by technology, all together, with transmission assumption)
nl = {}
for scenario in  ut.CESM2_REALIZATION_DICT: 
    simple = xr.open_dataset(f"{path}net_load_simple_{scenario}.nc").net_load 
    hydro = xr.open_dataset(f"{path}net_load_hydro_storage_{scenario}.nc").net_load_adjusted
    dispatched = hydro - simple
    dispatched["technology"] = "hydro_dispatched"
    by_tech = xr.open_dataset(f"{path}eng_vars_GWh_country_sum_{scenario}.nc").eng_vars
    by_tech = xr.concat([by_tech, dispatched],dim="technology")
    nl_scen = {
        "simple": simple,
        "hydro": hydro,
        "transmission": xr.open_dataset(f"{path}net_load_transmission_{scenario}.nc").net_load,
        "by_tech": by_tech,  
    }
    nl[scenario] = nl_scen

#list of different scenarios and assumptions
capacity_scenarios = nl[scenario]["transmission"].capacity_scenario.values
heating_scenarios = nl[scenario]["transmission"].heating_scenario.values
generation = ["PV", "Wind_onshore","Wind_offshore","hydro_ror","hydro_dispatched"]
demand = ["heating-demand", "cooling-demand","weather-insensitive_demand"]
technologies = ["PV", "Wind_onshore","Wind_offshore","hydro_ror","hydro_dispatched","heating-demand", "cooling-demand","weather-insensitive_demand"]
tech_tit = ["Generation", "Demand"]
scenario_configs = [["current","current_electrified"],["future","fully_electrified"],["future_wind_x2","current_electrified"]]

# get 90th percentile for all climate model data combined, but separate for each scenario config
qu_all = []
for x,capacity_scenario in enumerate(capacity_scenarios):
    qu_heat = []
    for j,heating_scenario in enumerate(heating_scenarios):
        nl_for_qu = []
        for scenario in nl.keys():
            nl_for_qu.append(nl[scenario]["transmission"].sel(capacity_scenario=capacity_scenario,heating_scenario=heating_scenario).dropna(dim='time').values)
        qu_heat.append(np.quantile(np.array(nl_for_qu),0.9))
    qu_all.append(qu_heat)
qu_all = xr.DataArray(
    qu_all,
    dims=["capacity_scenario","heating_scenario"],
    coords = {"capacity_scenario":capacity_scenarios,"heating_scenario":heating_scenarios}
)

# find mean values of load separated by tech, across climate periods
nl_tech_mn_scen = []
hydro_disp_mean = []
for scenario in ut.CESM2_REALIZATION_DICT:
    nl_tech_mn_scen.append(nl[scenario]["by_tech"].mean(dim=("time","member")))
# mean for all other technologies
nl_tech_mn_scen = xr.concat(nl_tech_mn_scen,dim="climate")
nl_tech_mn = nl_tech_mn_scen.mean("climate")

# =============
# === Fig 4 ===
# =============

heating_scenario ="current_electrified"
capacity_scenario = "current"

f,axes = plt.subplots(3,2,figsize=(7.2,6),sharey=True)  
season_name={"all":"All year","DJF":"Winter","JJA":"Summer"}
#preprocess
for m,season in enumerate(season_name):
    print(season)
    axs = axes[m]
    # get dataset of all transmission nl values above p90
    df_all = get_all_nl_above_p90(qu_all, nl,heating_scenario,capacity_scenario,season=season)
    for tech in technologies:
        total_sample = df_all[df_all["technology"]== tech]
        hist = total_sample[total_sample["scenario"] =="Historical climate"].value.values
        eoc = total_sample[total_sample["scenario"] =="End-of-century climate"].value.values
        print(tech, stats.ks_2samp(hist, eoc).pvalue)
    #tech plots
    plot_violin(axs,df_all)
    for x,tech_type in enumerate([generation,demand]):
        ax=axs[x]
        if x == 0:
            typ_dict = ut.generation_dict_line_break
        else:
            typ_dict = ut.demand_dict_line_break
        if m > 1:
            ax.set_xticks(np.arange(len(tech_type)),labels=[typ_dict[tech] for tech in tech_type],rotation=45)
        else:
            ax.set_xticks(np.arange(len(tech_type)),labels=["" for tech in tech_type],rotation=45)
        if m == 0:
            ax.set_title(tech_tit[x])
    axs[0].set_ylabel(season_name[season]) 
f.text(0.01,0.5,"Anomaly from climatology [GW]",rotation='vertical',verticalalignment='center', horizontalalignment='center',fontsize=12)
for i,a in enumerate(axes.flatten()):
    a.text(0.008,0.92,string.ascii_lowercase[i],weight="bold",transform=a.transAxes)
handles, labels = axs[1].get_legend_handles_labels() 
f.legend(handles, labels, loc='lower center',ncol=2,frameon=False)
plt.savefig(f"../../figs_CC_impacts/fig4.png",bbox_inches="tight", dpi = 600,transparent=True)

# ========================
# === Appendix fig 6-7 ===
# ========================

for j,config in enumerate(scenario_configs[1:]):
    f,axs = plt.subplots(1,2,figsize=(7.2,3),sharey=True)            
    # get dataset of all transmission nl values above p90
    df_all = get_all_nl_above_p90(qu_all, nl,config[1],config[0],season="all")
    #tech plots
    plot_violin(axs,df_all)
    for x,tech_type in enumerate([generation,demand]):
        ax=axs[x]
        if x == 0:
            typ_dict = ut.generation_dict_line_break
        else:
            typ_dict = ut.demand_dict_line_break
        ax.set_xticks(np.arange(len(tech_type)),labels=[typ_dict[tech] for tech in tech_type],rotation=45)
        ax.set_title(tech_tit[x])
    axs[0].set_ylabel("Anomaly from climatology [GW]")
    for i,a in enumerate(axs):
        a.text(0.008,0.92,string.ascii_lowercase[i],weight="bold",transform=a.transAxes)
    handles, labels = axs[1].get_legend_handles_labels() 
    f.legend(handles, labels, loc='lower center',ncol=2,frameon=False)
    plt.savefig(f"../../figs_CC_impacts/appendix_fig{6+j}.png",bbox_inches="tight", dpi = 600,transparent=True)
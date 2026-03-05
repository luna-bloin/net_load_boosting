import sys
sys.path.append("utils/")
import utils as ut
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import plot_config as pco
import string
from matplotlib.ticker import MaxNLocator

# ==================
# === Open files ===
# ==================

path = "/net/xenon/climphys/lbloin/energy_boost/"
nl = {}
for scenario in  ut.CESM2_REALIZATION_DICT:
    net_load_hydro = xr.open_dataset(f"{path}net_load_hydro_storage_{scenario}.nc").net_load_adjusted
    nl_transmission = xr.open_dataset(f"{path}net_load_transmission_{scenario}.nc").net_load
    nl_scen = {"hydro": net_load_hydro,"transmission":nl_transmission}
    nl[scenario] = nl_scen

# =========================
# === Appendix figure 3 ===
# =========================

member = "A"
scenario = 'historical'
scenario_configs = [["current","fully_electrified"],["future","fully_electrified"],["future_wind_x2","current_electrified"]]
year = slice("1995-02-09","1995-03-23")
lab = ["Copperplate assumption", "Simplified realistic transmission"]
f,axs=plt.subplots(1,3,figsize=(7.2,3),sharex=True,sharey=True)
for i,config in enumerate(scenario_configs):
    ax = axs[i]
    for n,typ in enumerate(["hydro","transmission"]):
        nl_here = nl[scenario][typ].sel(heating_scenario=config[1],capacity_scenario=config[0],time=year,member=member)
        nl_here.plot(color=pco.colors[1-n],linestyle="solid",label=lab[n],ax=ax)

for i,a in enumerate(axs.flatten()):
    pco.set_grid(a)
    a.text(0.005,0.93,string.ascii_lowercase[i],weight="bold",transform=a.transAxes)
    a.set_ylabel("")
    a.set_title("")
    a.set_xlabel("Time")
    a.set_xticks([-1780,-1752])
axs[0].set_ylabel("Net load [GWh]")
#f.text(0,0.5,"Net load [GWh]",rotation='vertical',verticalalignment='center', horizontalalignment='center')
    
handles, labels = axs[0].get_legend_handles_labels()
f.legend(handles, labels, loc='lower center',ncol=2)
f.suptitle(f"European net load, {scenario} climate")
plt.tight_layout()
plt.subplots_adjust(bottom=0.25)
plt.savefig(f"../../figs_CC_impacts/appendix_fig3.png",bbox_inches="tight", dpi = 600,transparent=True)
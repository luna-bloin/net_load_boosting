import sys
sys.path.append("../utils/")
# local imports
import plot_config as pco
import extreme_analysis as exa
#standard
import xarray as xr
#plotting
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
#other 
from datetime import timedelta
import string
import numpy as np

# =================
# === Functions ===
# =================

def get_atm_plots_top5(heat,capac,climate):
    # === get net load and top events for given heating, capacity scenario and climate period ===
    nl_one_scen = nl.sel(heating_scenario=heat,capacity_scenario=capac,climate=climate)
    extreme_here = extremes.sel(heating_scenario=heat,capacity_scenario=capac,climate=climate).dropna(dim="event",how="all")
    dur = extreme_here.sel(typ = "dur")
    cum = extreme_here.sel(typ="cum")
    cum = cum.sortby(cum,ascending=False)
    
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
    to_plot = (atms_mn_top5.groupby("time.dayofyear")-atm_mn)[["temperature","s_hub"]].mean("time")
    to_plot["Z500"] = atms_mn_top5["Z500"].mean("time")
    return to_plot

def plot_weather_maps(var,label,cmap,fsize=(7.2,3),cte=0):
    # plot temperature anomaly maps
    f=ds_to_plot[var].plot(
        x="lon",y="lat",col="scen",
        subplot_kws={"projection": ccrs.PlateCarree()},
        transform=ccrs.PlateCarree(),
        add_colorbar=False,
        figsize=fsize,
        cmap=cmap,
      )
    # plot z500 maps as contours
    levels = [ [round(int(ds_to_plot["Z500"].min()),-2) +x*100 for x in range(int((ds_to_plot["Z500"].max()-ds_to_plot["Z500"].min())/10)+1)], [round(int(ds_to_plot["Z500"].min()),-2) +x*25 for x in range(int((ds_to_plot["Z500"].max()-ds_to_plot["Z500"].min())/10)+1)]]
    for j, ax in enumerate(f.axs.flat):
        for n, level in enumerate(levels):
            z = ds_to_plot["Z500"].isel(scen=j)
            cs = ax.contour(
                z.lon,z.lat,z,
                colors="k",
                levels=level,
                linewidths = 0.7 - 0.4*n,
                alpha=1- 0.6*n,
            )
            if n ==0:
                ax.clabel(cs,inline=True,fontsize=10,levels=cs.levels, inline_spacing=3)
    
    # === fig configs ===
    # add borders and coastlines
    for i,a in enumerate(f.axs.flat):
        a = pco.add_country_borders(a)
        a.coastlines(linewidth=0.1)
        a.text(0.01,0.92,string.ascii_lowercase[i+cte],weight="bold",transform=a.transAxes)
    # make it have a tight layout (more difficult with cartopy)
    f.fig.subplots_adjust(left=0.08,right=0.88,bottom=0.02,top=0.98,wspace=0.05,hspace=0.02,)    
    cax = f.fig.add_axes([0.9, 0.1575, 0.035, 0.685])
    d = np.ceil(np.nanmax(np.abs(ds_to_plot[var].values)))
    f.fig.colorbar(
        f.axs.flat[0].collections[0],
        cax=cax,
        label=label,
        ticks=np.arange(-6,6+1,3),
    )
    f.set_titles("")
    f.fig.savefig(f"../../figs_storyline_extremes/weather_map_{var}.png",bbox_inches="tight",dpi=600,transparent=True)

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
# === Figure 5 ===
# ================

#open files in correct format
ds_elec = get_atm_plots_top5("fully_electrified","future","SSP370")
ds_curr = get_atm_plots_top5("current_electrified","future_wind_x2","SSP370")
ds_to_plot = xr.concat([ds_elec,ds_curr],dim="scen")

print("plot figure 5")
plot_weather_maps("temperature","Temperature anomaly[$^\circ$C]","RdBu_r")
plot_weather_maps("s_hub","Wind speed anomaly[m/s]","PuOr",cte=2)
print("figure saved")
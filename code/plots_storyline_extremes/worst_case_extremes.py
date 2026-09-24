import sys
sys.path.append("../utils/")
# local imports
import utils as ut
import plot_config as pco
import extreme_analysis as exa
#standard
import numpy as np
import pandas as pd
import xarray as xr
#plotting
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
#other 
import string
from datetime import timedelta
import random

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
        for event in (extremes_here.event):
            end = extremes_here.sel(typ="cum",event=event).time
            duration = extremes_here.sel(typ="dur",event=event)
            mem = extremes_here.member.sel(event=event).item()
            peak = exa.find_peak(nl_here,mem,end,duration)
            peak_nls.append({
                    "climate": climate.item(),
                    "Maximum net load [TW]": peak
                })
    return pd.DataFrame(peak_nls)

def find_cum_for_violins(extremes,heat,capac):
    """"get the duration and day of year of the end (as a pd dataframe, necessary for sns violinplot) for each energy shortfall event in the dataset extremes, for a given heating scenario heat and capacity scenario capac"""
    extreme_scen = extremes.sel(heating_scenario=heat,capacity_scenario=capac)
    cum_plot = extreme_scen.sel(typ="cum").dropna(dim="event",how="all").to_series().reset_index(name="Cumulative threshold"+ "\n"+  "exceedance [TWh]")
    return cum_plot

def find_dur_for_violins(extremes,heat,capac):
    """"get the duration of the energy drought (as a pd dataframe, necessary for sns violinplot) for each event in the dataset extremes, for a given heating scenario heat and capacity scenario capac"""
    extreme_scen = extremes.sel(heating_scenario=heat,capacity_scenario=capac)
    dur_plot = extreme_scen.sel(typ="dur").dropna(dim="event",how="all").to_series().reset_index(name="Duration [days]")
    return dur_plot

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
    ax.set_xticks([])
    return ax

def plot_doy_polar(nl,heat, capac,nl_qu,extremes,ax_doy,ax_cum,ax_dur):
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
        print_top_months(doy,climate)
        theta = 2 * np.pi * (doy - 1) / 365
        for j,cum_ext in enumerate(cum[0:5]):
            #plot top five events in terms of maximum net load and duration
            nl_ext = exa.find_peak(nl_one_scen,cum_ext.member,cum_ext.time,dur.sel(time=cum_ext.time))
            for a in [ax_cum,ax_dur]:
                lm = a.get_xlim()
                a.plot((-1+x*2+random.uniform(-0.9, 0.9))*(lm[1]-lm[0])*0.015,cum.sel(time=cum_ext.time),"o",markersize=3.5,zorder=4,color=pco.colors[1-x],mec="k",mew=0.7)
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
    return ax_doy, ax_cum,ax_dur

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
    to_plot = atms_mn_top5[["Z500","s_hub"]].mean("time")
    to_plot["temperature"] = (atms_mn_top5.groupby("time.dayofyear")-atm_mn)["temperature"].mean("time")
    return to_plot

def plot_weather_maps(ax,ds_to_plot,cmap,vmin,vmax,zmin,zmax,var='temperature'):
    # plot temperature anomaly maps
    f=ds_to_plot[var].plot(
        #subplot_kws={"projection": ccrs.PlateCarree()},
        transform=ccrs.PlateCarree(),
        add_colorbar=False,
        cmap=cmap,
        ax=ax,
        vmin= vmin,
        vmax=vmax,
      )
    # plot z500 maps as contours
    for n,level_type in enumerate([100,25]):
        level = [round(int(zmin),-2) +x*level_type for x in range(int((zmax-zmin)/10)+1)]
        z = ds_to_plot["Z500"]
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
    ax = pco.add_country_borders(ax)
    ax.coastlines(linewidth=0.1)
    return f,ax

def print_top_months(doy,climate):
    # Convert DOY to Month name (assuming a standard 2023 non-leap year)
    df = pd.DataFrame({'DOY': doy})
    df['Date'] = pd.to_datetime(df['DOY'], format='%j')
    df['Month'] = df['Date'].dt.month_name()
    
    
    # --- Top Months containing 90% of Data ---
    counts = df['Month'].value_counts()
    cum_pct = counts.cumsum() / len(df)
    top_90_months = cum_pct[cum_pct.shift(1, fill_value=0) < 0.9].index.tolist()
    
    print(f"Top months holding 90% of data in {climate}: {top_90_months}")
# ===============================
# === Opening the right files ===
# ===============================

print("opening files")
path = "/net/xenon/climphys/lbloin/energy_boost/"
nl, nl_qu,extremes = exa.open_all_parent_nl(path,nl_only=True)
capacity_scenarios = nl.capacity_scenario.values
heating_scenarios = nl.heating_scenario.values
# open atmospheric variables
atm,atm_mn = exa.open_atm_vars(path)
print("files opened")

# ================
# === Figure 1 ===
# ================
print("plotting figure 1")
fig = plt.figure(figsize=(7.2,4.5))
gs = fig.add_gridspec(nrows=2,ncols=3,width_ratios=[1,1,1.5],wspace=0.2,hspace=0.2,)

two_representative_scenarios = [["future","fully_electrified"],["future_wind_x2","current_electrified"]]

#concat weather maps to get min max values for colorbar and isolines
ds_elec = get_atm_plots_top5(two_representative_scenarios[0][1],two_representative_scenarios[0][0],"SSP370")
ds_curr = get_atm_plots_top5(two_representative_scenarios[1][1],two_representative_scenarios[1][0],"SSP370")
ds_to_plot = xr.concat([ds_elec,ds_curr],dim="scen")
abs_val = np.abs(ds_to_plot.temperature).max()
max_val_s_hub = ds_to_plot.s_hub.max()

for i,scen in enumerate(two_representative_scenarios):
    heat = scen[1]
    capac = scen[0]
    # set up figure
    ax_cum = fig.add_subplot(gs[i, 0])
    ax_doy = fig.add_subplot(gs[i, 1], projection="polar")
    ax_map = fig.add_subplot(gs[i, 2],projection=ccrs.PlateCarree())
    f_dur,ax_dur=plt.subplots()
    #plot cumulative threshold exceedance violin plots for each event
    cum_plot = find_cum_for_violins(extremes,heat,capac)    
    ax_cum = plot_violin_clim_halves(cum_plot,ax_cum,0)     
    # get seasonality data and plot it
    ax_doy,ax_cum,ax_dur = plot_doy_polar(nl,heat, capac,nl_qu,extremes,ax_doy,ax_cum,ax_dur)       

    # plot maps of top 5 events
    to_plot_here = ds_to_plot.isel(scen=i)
    plot_weather_maps(ax_map,to_plot_here,"RdBu_r",-abs_val,abs_val,ds_to_plot.Z500.min(),ds_to_plot.Z500.max())

# Figure configs
axes = fig.get_axes()
# letter labels (a,b,c...)
for i,a in enumerate(axes):
    if i ==1 or i==4:
        a.text(0.01,0.97,string.ascii_lowercase[i],weight="bold",transform=a.transAxes)
    else:
        a.text(0.01,0.92,string.ascii_lowercase[i],weight="bold",transform=a.transAxes)

#legend for first two columns
handles, labels = ax_doy.get_legend_handles_labels() 
fig.legend(handles, labels, loc='lower left',ncol=2,frameon=False,bbox_to_anchor=(0.04, -0.073),)

# colorbar for last column
pos = gs[1, 2].get_position(fig)
cax = fig.add_axes([
    pos.x0,       # left
    0.05,         # bottom
    pos.width,    # same width as map column
    0.025,        # height
])
fig.colorbar(
        ax_map.collections[0],
        cax=cax,
        label="Temperature anomaly[$^\circ$C]",
        orientation="horizontal",
    )
fig.savefig(f"../../figs_storyline_extremes/spa_season_map.png",bbox_inches="tight",dpi=1200,transparent=True)
print("figure 1 saved")

# =========================
# === Appendix figure 6 ===
# =========================

print("plotting appendix figure 6")
# plot abs values of wind speeds
f,ax=plt.subplots(1,2,figsize=(7.2,3),subplot_kw={'projection':ccrs.PlateCarree()})
for i,scen in enumerate(two_representative_scenarios):
    to_plot_here = ds_to_plot.isel(scen=i)
    plot_weather_maps(ax[i],to_plot_here,"cividis",0,max_val_s_hub,ds_to_plot.Z500.min(),ds_to_plot.Z500.max(),var='s_hub')
    ax[i].text(0.01,0.92,string.ascii_lowercase[i],weight="bold",transform=ax[i].transAxes,color="white")
    ax[i].set_title("")
# make it have a tight layout (more difficult with cartopy)
f.subplots_adjust(left=0.08,right=0.88,bottom=0.02,top=0.98,wspace=0.05,hspace=0.02,)    
cax = f.add_axes([0.9, 0.1575, 0.015, 0.685])
f.colorbar(
    ax[0].collections[0],
    cax=cax,
    label="Wind speed [m/s]",
)
f.savefig(f"../../figs_storyline_extremes/wind_maps_WCED.png",bbox_inches="tight",dpi=1200,transparent=True)
print("appendix figure 6 saved")

## =============================
# === Appendix figures 2-3,6 ===
# ==============================

print("plotting appendix figures 2-3+6")
f_cum,ax_cum = plt.subplots(2,4,figsize=(7.2,4),sharey=True) #plot cumulative threshold exceedance for all scenarios 
f_dur,ax_dur = plt.subplots(2,4,figsize=(7.2,4),sharey=True) #plot duration for all scenarios 

f_doy,ax_doy = plt.subplots(2,4,figsize=(7.2,4),subplot_kw={"projection":"polar"}) # plot seasonality (day of year of event end) for all scenarios

# iterate over all scenarios
for i,heat in enumerate(heating_scenarios):
    for j,capac in enumerate(capacity_scenarios):
        print(heat, capac)
        # plot cumulative threshold exceedance  violin plots for each event
        cum_plot = find_cum_for_violins(extremes,heat,capac)    
        ax_cum[i][j] = plot_violin_clim_halves(cum_plot,ax_cum[i][j],0)     
        # get seasonality data and plot it
        ax_doy[i][j],ax_cum[i][j],ax_dur[i][j] = plot_doy_polar(nl,heat, capac,nl_qu,extremes,ax_doy[i][j],ax_cum[i][j],ax_dur[i][j])
        # plot maximum net load and duration violin plots for each event
        dur_plot = find_dur_for_violins(extremes,heat,capac)    
        ax_dur[i][j] = plot_violin_clim_halves(dur_plot,ax_dur[i][j],0)     
        
#fig config
for i, a in enumerate([ax_cum,ax_doy,ax_dur]):
    for n,ax in enumerate(a.flatten()):
        if i < 2: #only for intensity and duration plot
            ax.set_xticks([])
            ax.set_ylabel("")
        ax.text(0.01,0.97,string.ascii_lowercase[n],weight="bold",transform=ax.transAxes)

f_cum.text(0.01,0.5,"Cumulative threshold exceedance [TWh]",rotation='vertical',verticalalignment='center', horizontalalignment='center')
f_dur.text(0.01,0.5,"Duration [days]",rotation='vertical',verticalalignment='center', horizontalalignment='center')

typ = ["SPA","seasonality","duration"]
for i,fig in enumerate([f_cum,f_doy,f_dur]):
    handles, labels = ax_doy[0][0].get_legend_handles_labels() 
    fig.legend(handles, labels, loc='lower center',ncol=5,frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.1)
    fig.savefig(f"../../figs_storyline_extremes/{typ[i]}_all_scenarios.svg",bbox_inches="tight",dpi=600,transparent=True)

print("appendix figures 2-3+6 saved")

# ============================
# === Appendix figures 4-5 ===
# ============================

# iterate over all scenarios
for x, climate in enumerate(["historical", "SSP370"]):
    ds_to_plot = []
    print(climate)
    #concat weather maps to get min max values for colorbar and isolines
    for i,heat in enumerate(heating_scenarios):
        for j,capac in enumerate(capacity_scenarios):
            ds_to_plot.append(get_atm_plots_top5(heat,capac,climate))
    ds_to_plot = xr.concat(ds_to_plot,dim="scen")
    zmin=ds_to_plot.Z500.min()
    zmax=ds_to_plot.Z500.max()
    
    # plot temperature anomaly maps
    f=ds_to_plot['temperature'].plot(
        x="lon",y="lat",col="scen",col_wrap=4,
        subplot_kws={"projection": ccrs.PlateCarree()},
        transform=ccrs.PlateCarree(),
        figsize=(8.5,4),
        cmap="RdBu_r",
        cbar_kwargs={"shrink":0.8,"label":"Temperature anomaly[$^\circ$C]"},
      )
    # plot z500 maps as contours
    for j,scen in enumerate(ds_to_plot.scen):
        for n,level_type in enumerate([200,50]):
            z = ds_to_plot.isel(scen=j).Z500
            ax=f.axs.flat[j]
            level = [round(int(zmin),-2) +x*level_type for x in range(int((zmax-zmin)/10)+1)]
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
            ax = pco.add_country_borders(ax)
            ax.coastlines(linewidth=0.1)
            ax.set_title("")
    f.fig.savefig(f"../../figs_storyline_extremes/weather_WCED_{climate}.png",bbox_inches="tight",dpi=1200,transparent=True)






import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
sys.path.append("../utils/")
# local imports
import utils as ut
import plot_config as pco
import extreme_analysis as exa



# ===========================
# === Load the shapefiles ===
# ===========================
print("opening shapefiles")
shapefile_path = "../../inputs/geopandas/ne_50m_admin_0_countries.shp"
world = gpd.read_file(shapefile_path)
europe = world[world['CONTINENT'] == 'Europe'] # filter for Europe
#offshore zones
eez_gdf = gpd.read_file("../../../../CESM2energy/inputs/EEZ/eez_v11.shp")
print("shapefiles open")


# ========================================================
# === Get on and offshore data to plot as one big list ===
# ========================================================
capacities = xr.open_dataset("/net/xenon/climphys/lbloin/energy_boost/installed_capacity_scenarios.nc")
geopoandas_list = []
for j,capac in enumerate(["future","future_wind_x2"]):
    capacity = capacities.sel(capacity_scenario=capac).GWh
    # open data as geopandas data
    #offshore wind
    geopoandas_list.append(eez_gdf.merge(capacity.sel(technology="Wind_offshore").to_dataframe(name="capacity"), left_on="SOVEREIGN1", right_on="country")[["SOVEREIGN1", "capacity", "geometry"]])
    #onshore wind
    geopoandas_list.append(europe.merge(capacity.sel(technology="Wind_onshore").to_dataframe(name="capacity"), left_on="NAME_LONG", right_on="country")[["NAME_LONG", "capacity", "geometry"]])



# ==============================
# === Plot appendix figure 7 ===
# ==============================

# fig configs
fig, axs = plt.subplots(1,2,figsize=(7.2, 3),sharex=True,sharey=True)
cax = fig.add_axes([0.92, 0.13, 0.015, 0.73]) # [left, bottom, width, height]
tit_list = ["Offshore wind","Onshore wind","Solar PV"]
vmax = np.max([gp_data.capacity.max() for gp_data in geopoandas_list]) # max value (for colorbar)

print("plotting appendix figure 6")
for i, ds in enumerate(geopoandas_list):
    ax = axs[int(i/2)] #so that on and offshore in one scneario is plotted in the same figure
    lgd = False
    if i == 3:
        lgd = True
    #plot
    ds.plot(
        column="capacity",
        cmap="viridis",
        legend=lgd,
        cax=cax if lgd else None, # Direct colorbar to dedicated axis
        legend_kwds={"label": "Installed capacity [GW]"} if lgd else None,
        edgecolor="black",
        linewidth=0.3,
        ax=ax,
        vmin=0,
        vmax=vmax,
    )
    # Turn off x and y axis ticks, labels, and outer frame
    ax.set_xticks([])
    ax.set_yticks([])
    # Set map limits to same as weather maps
    ax.set_xlim([-15, 32])
    ax.set_ylim([30, 75])
    ax.set_title("", fontsize=10)
plt.savefig(f"../../figs_storyline_extremes/installed_wind_capacity.png",bbox_inches="tight",dpi=600,transparent=True)
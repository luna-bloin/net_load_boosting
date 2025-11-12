import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
import xarray as xr

# the colors in this colormap come from http://colorbrewer2.org
# the 8 more saturated colors from the 9 blues / 9 reds
cmap = ListedColormap([
    '#08306b', '#08519c', '#2171b5', '#4292c6',
    '#6baed6', '#9ecae1', '#c6dbef', '#deebf7',
    '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a',
    '#ef3b2c', '#cb181d', '#a50f15', '#67000d',
])

# === Functions ===
def preproc_ds(clim_period):
    # open and process data to get yearly mean temperatures
    ds = xr.open_dataset(f"/net/xenon/climphys/lbloin/energy_boost/bced_atm_vars_{clim_period}.nc").temperature.mean(("lat","lon")).resample(time="1YE").mean().isel(time=slice(None,-2))
    ds_stacked = ds.stack(dim=("time","member"))
    ds_mean = ds_stacked.mean() 
    return ds_stacked, ds_mean

def plot_stripes(anom,clim_period,cmap,LIM = 1.5):
    """plots yearly temperature anomalies as warming stripes"""
    fig = plt.figure(figsize=(10, 1))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    
    col = PatchCollection([
        Rectangle((y, 0), 1, 1)
        for y in range(len(anom.dim))
    ])
    
    # set data, colormap and color limits
    
    col.set_array(anom)
    col.set_cmap(cmap)
    col.set_clim(-LIM,LIM)
    ax.add_collection(col)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 57)
    
    fig.savefig(f'../../figs_CC_impacts/warming-stripes_{clim_period}.png',dpi=1200,transparent=True)


# === Open files and preprocess === 
ds_hist,ds_mean = preproc_ds("historical")
ds_ssp370,ds_mean_future = preproc_ds("SSP370")
#both temperature anomalies are calculated with the same reference - the historical mean
anom_hist = ds_hist-ds_mean
anom_ssp370 = ds_ssp370-ds_mean

# === Plot for both climate periods ===
plot_stripes(anom_hist,"historical",cmap,LIM = 3)
plot_stripes(anom_ssp370,"SSP370",cmap,LIM = 3)
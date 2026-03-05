# the postprocessing concatenates the parent and boosted run, to ensure continuity
import xarray as xr
import cftime
from datetime import timedelta
import sys

try: 
    dates = sys.argv[1:5]
    mems = eval(sys.argv[6])
    scenario = sys.argv[7]
    realization = sys.argv[8]
    nb_realization = sys.argv[9]
except:
    dates = ["2088-12-02","2088-12-05","2088-12-08","2088-12-11"] #["2081-12-26", "2081-12-29","2082-01-01", "2082-01-04"] #["2080-02-14","2080-02-16","2080-02-18","2080-12-01","2080-12-03","2080-12-05"]  
    mems = 20
    scenario = "SSP370"
    realization = "A"
    nb_realization = 1500

# functions
def str_to_cftime_noleap(string):
    return cftime.DatetimeNoLeap(int(string[0:4]), int(string[5:7]), int(string[8:10]))

# in and output paths
input_path = "/net/meso/climphys/cesm212/boosting/archive/"
parent_path = f"/net/meso/climphys/cesm212/b.e212.B{scenario}cmip6.f09_g17.{nb_realization}/archive/atm/hist/"
output_path = f"/net/xenon/climphys/lbloin/CESM2energy/output/boost/{realization}/atmospheric_variables/"
    
for date in dates:
    print(date)
    # find the starting date for concatenation: we want 1 week before boosting starts:
    start_date = str_to_cftime_noleap(date) - timedelta(days=7)
    if start_date.year != int(date[0:4]): # what if 1 week before the boosting happens in a different year?
        print("need to concat more than one year of parent files")
        ds_parent_first_year = xr.open_dataset(f"{parent_path}b.e212.B{scenario}cmip6.f09_g17.{nb_realization}.cam.h6.{start_date.year}-01-01-03600.nc").isel(lev=slice(29, 32), ilev=slice(29, 33))
        ds_parent_second_year = xr.open_dataset(f"{parent_path}b.e212.B{scenario}cmip6.f09_g17.{nb_realization}.cam.h6.{date[0:4]}-01-01-03600.nc").isel(lev=slice(29, 32), ilev=slice(29, 33))
        ds_parent = xr.concat([ds_parent_first_year.sel(time=slice(start_date,None)),ds_parent_second_year.sel(time=slice(None,date))],dim="time").isel(time=slice(0,-23))
    else:
        ds_parent = xr.open_dataset(f"{parent_path}b.e212.B{scenario}cmip6.f09_g17.{nb_realization}.cam.h6.{date[0:4]}-01-01-03600.nc").isel(lev=slice(29, 32), ilev=slice(29, 33))
        ds_parent = ds_parent.sel(time=slice(start_date,date)).isel(time=slice(0,-23)) #select time up to the hour before boosting happens
    for mem in range(1,mems+1):
        print(mem)
        #atmospheric variables
        ds_boost=xr.open_dataset(f"{input_path}B{scenario}cmip6.000{nb_realization}.{date}.ens{mem:03d}/atm/hist/B{scenario}cmip6.000{nb_realization}.{date}.ens{mem:03d}.cam.h6.{date}-03600.nc").isel(lev=slice(29, 32), ilev=slice(29, 33))
        xr.concat([ds_parent,ds_boost],dim="time").to_netcdf(f"{output_path}atmospheric_variables_{date}_ens{mem:03d}.nc")
        

        
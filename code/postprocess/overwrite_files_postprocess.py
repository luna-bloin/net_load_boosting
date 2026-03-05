# overwrites the expensive boosted files to only include the small version
import xarray as xr
import cftime
from datetime import timedelta
import sys

try: 
    dates = sys.argv[1:5]
    mems = eval(sys.argv[5])
    scenario = sys.argv[6]
    realization = sys.argv[7]
    nb_realization = sys.argv[8]
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
output_path = f"/net/xenon/climphys/lbloin/CESM2energy/output/boost/{realization}/atmospheric_variables/"

for date in dates:
    print(date)
    # find the starting date for concatenation: we want 1 week before boosting starts:
    start_date = str_to_cftime_noleap(date) - timedelta(days=7)
    for mem in range(1,mems+1):
        print(mem)
        ds_boost = xr.open_dataset(f"{output_path}atmospheric_variables_{date}_ens{mem:03d}.nc")
        ds_boost.to_netcdf(f"{input_path}B{scenario}cmip6.000{nb_realization}.{date}.ens{mem:03d}/atm/hist/B{scenario}cmip6.000{nb_realization}.{date}.ens{mem:03d}.cam.h6.{date}-03600.nc")

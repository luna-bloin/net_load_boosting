# This py script exists to postprocess the boosted runs of B 1996, since the original files were corrupted. all other boosted postprocessing can be found in the script postprocess_boosted_CESM2.py
# This processing script should no longer be executed, but exists rather as a documentation of what was done
# the postprocessing concatenates the parent and boosted run, to ensure continuity

import xarray as xr
import sys
sys.path.append("../utils")
import utils as ut

def select_Europe(ds):
    return ds.sel(lon=slice(-15, 50), lat=slice(30, 75))

dates = ["1996-01-20","1996-01-23","1996-01-26","1996-01-29","1996-02-01"]
mems = 50

input_location = "/net/xenon/climphys/lbloin/CESM2energy/output/"

#types of variables to postprocess
bced_vars = ["discharge","global-horizontal","s100","temperature"]
secondary_vars = ["alpha", "other", "rho"] 

for date in dates:
    print(date)
    #parent file for "other"
    input_path_parent = f"/net/meso/climphys/cesm212/b.e212.BHISTcmip6.f09_g17.1000/archive/atm/hist/b.e212.BHISTcmip6.f09_g17.1000.cam.h6.{date[0:4]}-01-01-03600.nc"
    parent_ds_other = select_Europe(ut.zero_mean_longitudes(xr.open_dataset(input_path_parent)[["U10","QREFHT"]])).sel(time=slice(None,date))
    for mem in range(1,mems+1):
        print(mem)
        for i, var_type in enumerate([bced_vars,secondary_vars]):
            if i ==0:
                bced = "bced_"
            else:
                bced = ""
            for var in var_type:
                #open parent file, select january 1st until boost date
                if var == "other": # non-boosted runs do not have this saved, so we need to create it from the parent run
                    parent_ds = parent_ds_other
                else:
                    parent_ds = xr.open_dataset(f"{input_location}bias_correction/B/historical/B/atmospheric_variables/{bced}CESM2_{var}_{date[0:4]}.nc").sel(time=slice(None,date))
                if var != "discharge": #discharge has daily resolution
                    parent_ds = parent_ds.isel(time=slice(0,-23)) #make sure to have exactly right number of hours of boosting day
                boost_ds = xr.open_dataset(f"{input_location}boost/B/atmospheric_variables/{bced}{var}_boost_{date}_ens{mem:03d}.nc")
                xr.concat([parent_ds,boost_ds],dim="time").to_netcdf(f"{input_location}boost/B/atmospheric_variables/{bced}{var}_boost_{date}_ens{mem:03d}_concatenated.nc")
    
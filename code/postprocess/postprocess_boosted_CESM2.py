import xarray as xr

try: 
    dates = sys.argv[1]
    mems = sys.argv[2]
except:
    dates = ["1996-01-20","1996-01-23","1996-01-26","1996-01-29","1996-02-01"]
    mems = 50
    
for date in dates:
    print(date)
    for mem in range(1,mems+1):
        print(mem)
        file = f"/net/meso/climphys/cesm212/boosting/archive/BHISTcmip6.1001000.{date}.ens{mem:03d}/atm/hist/BHISTcmip6.1001000.{date}.ens{mem:03d}.cam.h6.{date}-03600.nc"
        ds = xr.open_dataset(file).isel(lev=slice(29, 32), ilev=slice(29, 33))
        ds.to_netcdf(file)
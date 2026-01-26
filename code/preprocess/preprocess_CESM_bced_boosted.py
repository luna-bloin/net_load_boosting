import sys
sys.path.append("../utils")
import utils as ut
import preprocess as pc
import bias_correct_funcs as bc
import pandas as pd
import xarray as xr
from tqdm import tqdm
import glob
import subprocess
from bias_correction import BiasCorrection


# ======================================================================================================================================================================================================================
# === Preprocesses CESM2  before climate2energy conversion into nc files: TREFHT, FSDS, S100 and discharge are already bias corrected by the tool, and this data is read. For Z500, bias correction is also applied. ===
# ======================================================================================================================================================================================================================


out_path = "/net/xenon/climphys/lbloin/energy_boost/"
atm_vars = {"temperature":"temperature", "global-horizontal":"global_horizontal","s100":"s_hub"}

if __name__ == "__main__":    
    boost_date = sys.argv[1] #date of boosting
    realization = sys.argv[2]
    scenario = sys.argv[3]
    in_path = f"/net/xenon/climphys/lbloin/CESM2energy/output/boost/{realization}/"
    
    # === atmospheric variables: temperature, global horizontal, wind speed (s_hub) ===
    dss = []
    for var in atm_vars:
        print(var)
        files = glob.glob(f"{in_path}/atmospheric_variables/bced_{var}_boost_{boost_date}_ens*.nc")
        ds_mem = xr.open_mfdataset(files,concat_dim="member", combine="nested")[atm_vars[var]].convert_calendar("noleap").resample(time="1D").mean()
        ds_mem["member"] = range(1,len(ds_mem.member)+1)
        dss.append(ds_mem)        
    dss = xr.Dataset({da.name: da for da in dss})
    dss.to_netcdf(f"{out_path}bced_atm_vars_boost_{realization}_{boost_date}.nc")
    
    # === river discharge (different grid, so needs to be in a separate file ===
    ds_mem = []
    files = glob.glob(f"{in_path}/atmospheric_variables/bced_discharge_boost_{boost_date}_ens*.nc")
    ds_discharge=xr.open_mfdataset(files,concat_dim="member", combine="nested")["discharge"].convert_calendar("noleap")
    ds_discharge["member"] = range(1,len(ds_discharge.member)+1)
    ds_discharge.to_netcdf(f"{out_path}bced_discharge_boost_{realization}_{boost_date}.nc")

    # === Z500 ===
    # Get ERA5 data
    path = f"{out_path}Raw_ERA5_z500.nc"
    if glob.glob(path) == []:
        subprocess.run(["bash", f"preprocess/preprocess_Z500_ERA5.sh"])
    z500_ERA5 = xr.open_dataset(path).resample(time="1D").mean().convert_calendar("noleap").drop_vars("level")

    # get boosted z500 data, and bias correct it with ERA5
    # Bias correction is done AA, BB, CC (cesm model realization used for bias correction is the same as the data that is corrected)
    # open raw historical and ssp370 z500
    hist_z500 = pc.preproc_cesm_z500("historical",realization)
    boosted_z500 = pc.preproc_cesm_z500_boosted(scenario,realization,boost_date)
    # perform bias correction for each scenario considered (with hist_z500 as the model reference)
    z500_bc = bc.bias_correct_dataset(boosted_z500.Z500, hist_z500.Z500, z500_ERA5.Z500)
    z500_bc.to_netcdf(f"{out_path}bced_z500_{scenario}_boost_{realization}_{boost_date}.nc")
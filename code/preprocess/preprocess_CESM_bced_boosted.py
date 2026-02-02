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

# =======================================================================================================================================================================================================================
# === Preprocesses CESM2 atmospheric data into maps over Europe: for TREFHT, FSDS, S100, Z500, bias correction using ERA5 is applied. For discharge, the already bias corrected data from climate2energy is retrieved ===
# =======================================================================================================================================================================================================================

# =============
# === paths ===
# =============

in_path_discharge = "/net/xenon/climphys/lbloin/CESM2energy/output/boost/"
out_path = "/net/xenon/climphys/lbloin/energy_boost/"

boost_date = sys.argv[1] #date of boosting
realization = sys.argv[2]
scenario = sys.argv[3]

# =================
# === Discharge ===
# =================

print("getting bias corrected discharge data")
# === river discharge (different grid, so needs to be in a separate file ===
ds_mem = []
files = sorted(glob.glob(f"{in_path_discharge}{realization}/atmospheric_variables/bced_discharge_boost_{boost_date}_ens*.nc"))
ds_discharge=xr.open_mfdataset(files,concat_dim="member", combine="nested")["discharge"].convert_calendar("noleap")
ds_discharge["member"] = range(1,len(ds_discharge.member)+1)
ds_discharge.to_netcdf(f"{out_path}bced_discharge_boost_{realization}_{boost_date}.nc")
print("Discharge done")

# ===========================
# === All other variables ===
# ===========================

atm_vars = {"temperature":"temperature", "global-horizontal":"global_horizontal","s100":"s_hub","Z500":"Z500"}

print("getting atmospheric data")
for var in atm_vars:
    print(var)
    # === Step 1: get ERA5 data for bias correction ===
    print("Get ERA5 data")
    era5_file = f"{out_path}Raw_ERA5_{var}.nc"
    if glob.glob(era5_file) == []:
        subprocess.run(["bash", f"ERA5_preproc_scripts/preprocess_{var}_ERA5.sh"])
    ERA5 = xr.open_dataset(era5_file).convert_calendar("noleap")
    if var =="Z500":
        ERA5 = ERA5.drop_vars("level")
    print("ERA5 done")
    # === Step 2: get boosted data and bias correct it
    # Bias correction is done AA, BB, CC (cesm2 model realization used for bias correction is the same as the data that is corrected)
    # open raw boosted and historical
    hist = pc.preproc_cesm2("historical",realization,out_path,var)
    boosted = pc.preproc_cesm2_boosted(boost_date,scenario,realization,out_path,var)
    # perform bias correction and save
    bc_ds = bc.bias_correct_dataset(boosted[atm_vars[var]], hist[atm_vars[var]], ERA5[atm_vars[var]])
    bc_ds.to_netcdf(f"{out_path}bced_{var}_{scenario}_boost_{realization}_{boost_date}.nc")
print("atmospheric data done")

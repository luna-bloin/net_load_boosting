import sys
sys.path.append("../utils")
import utils as ut
import preprocess_atm_fields as pc
import bias_correct_funcs as bc
import pandas as pd
import xarray as xr
from tqdm import tqdm
import glob
import subprocess
from bias_correction import BiasCorrection


# =======================================================================================================================================================================================================================
# === Preprocesses CESM2 atmospheric data into maps over Europe: for TREFHT, FSDS, S100, Z500, bias correction using ERA5 is applied. For discharge, the already bias corrected data from climate2energy is retrieved ===
# =======================================================================================================================================================================================================================

# =============
# === paths ===
# =============

in_path_discharge = "/net/xenon/climphys/lbloin/CESM2energy/output/bias_correction/"
out_path = "/net/xenon/climphys/lbloin/energy_boost/"

# =================
# === Discharge ===
# =================

# print("getting bias corrected discharge data")
# for scenario in ut.CESM2_REALIZATION_DICT:
#     members = list(ut.CESM2_REALIZATION_DICT[scenario].keys())    
#     ds_mem = []
#     for mem in members:
#         ds_mem.append(xr.open_mfdataset(f"{in_path_discharge}{mem}/{scenario}/{mem}/atmospheric_variables/bced_CESM2_discharge_*.nc")["discharge"].convert_calendar("noleap"))
#     ds_discharge = xr.concat(ds_mem,pd.Index(members, name="member"))
#     ds_discharge.to_netcdf(f"{out_path}bced_discharge_{scenario}.nc")
# print("Discharge done")

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
    # get CESM2 historical and SSP370, and bias correct it with ERA5
    # Bias correction is done AA, BB, CC (cesm2 model realization used for bias correction is the same as the data that is corrected)
    members=["A","B","C"]
    for member in members:
        # open raw historical and ssp370
        hist = pc.preproc_cesm2("historical",member,out_path,var)
        ssp = pc.preproc_cesm2("SSP370",member,out_path,var)
        ds_scenario = {"historical":hist,"SSP370":ssp}
        # perform bias correction for each scenario considered (with hist as the model reference)
        for scenario in tqdm(ds_scenario):
            bc_ds = bc.bias_correct_dataset(ds_scenario[scenario][atm_vars[var]], ds_scenario["historical"][atm_vars[var]], ERA5[atm_vars[var]])
            bc_ds.to_netcdf(f"{out_path}bced_{atm_vars[var]}_{scenario}_{member}.nc")
    
    # concatenate all members together to one file per scenario
    for scenario in ut.CESM2_REALIZATION_DICT:
        file = sorted(glob.glob(f"{out_path}bced_{atm_vars[var]}_{scenario}_*.nc"))
        ds=xr.open_mfdataset(file,concat_dim="member",combine="nested")
        ds["member"] = members
        ds.to_netcdf(f"{out_path}bced_{atm_vars[var]}_{scenario}.nc")
print("atmospheric data done")
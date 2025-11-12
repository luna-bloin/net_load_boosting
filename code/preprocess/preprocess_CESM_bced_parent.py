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

in_path = "/net/xenon/climphys/lbloin/CESM2energy/output/bias_correction/"
out_path = "/net/xenon/climphys/lbloin/energy_boost/"
atm_vars = {"temperature":"temperature", "global-horizontal":"global_horizontal","s100":"s_hub"}

for scenario in ut.CESM2_REALIZATION_DICT:
    members = list(ut.CESM2_REALIZATION_DICT[scenario].keys())
    # === atmospheric variables: temperature, global horizontal, wind speed (s_hub) ===
    dss = []
    for var in atm_vars:
        ds_mem = []
        for mem in members:
            ds_mem.append(xr.open_mfdataset(f"{in_path}{mem}/{scenario}/{mem}/atmospheric_variables/bced_CESM2_{var}_*.nc")[atm_vars[var]].convert_calendar("noleap"))
        dss.append(xr.concat(ds_mem,pd.Index(members, name="member")))        
    dss.to_netcdf(f"{out_path}bced_atm_vars_{scenario}.nc")
    
    # === river discharge (different grid, so needs to be in a separate file ===
    ds_mem = []
    for mem in members:
        ds_mem.append(xr.open_mfdataset(f"{in_path}{mem}/{scenario}/{mem}/atmospheric_variables/bced_CESM2_discharge_*.nc")["discharge"].convert_calendar("noleap"))
    ds_discharge = xr.concat(ds_mem,pd.Index(members, name="member"))
    ds_discharge.to_netcdf(f"{out_path}bced_discharge_{scenario}.nc")

# === Z500 ===
# Get ERA5 data
path = f"{out_path}Raw_ERA5_z500.nc"
if glob.glob(path) == []:
    subprocess.run(["bash", f"preprocess/preprocess_Z500_ERA5.sh"])
z500_ERA5 = xr.open_dataset(path).resample(time="1D").mean().convert_calendar("noleap").drop_vars("level")

# get CESM historical and SSP370, and bias correct it with ERA5
# Bias correction is done AA, BB, CC (cesm model realization used for bias correction is the same as the data that is corrected)
members=["A","B","C"]
z500_mem = []
for member in members:
    # open raw historical and ssp370 z500
    hist_z500 = pc.preproc_cesm_z500("historical",member)
    ssp_z500 = pc.preproc_cesm_z500("SSP370",member)
    z500_scenario = {"historical":hist_z500,"SSP370":ssp_z500}
    # perform bias correction for each scenario considered (with hist_z500 as the model reference)
    for scenario in tqdm(z500_scenario):
        z500_bc = bc.bias_correct_dataset(z500_scenario[scenario].Z500, z500_scenario["historical"].Z500, z500_ERA5.Z500)
        z500_bc.to_netcdf(f"{out_path}bced_z500_{scenario}_{member}.nc")

# concatenate all members together to one file per scenario
for scenario in ut.CESM2_REALIZATION_DICT:
    file = sorted(glob.glob(f"{out_path}bced_z500_{scenario}_*.nc"))
    ds=xr.open_mfdataset(file,concat_dim="member",combine="nested")
    ds["member"] = members
    ds.to_netcdf(f"{out_path}bced_z500_{scenario}.nc")

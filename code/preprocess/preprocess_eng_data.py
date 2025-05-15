import sys
sys.path.append("../utils")
import utils as ut
import preprocess as pc
import pandas as pd
import xarray as xr
from tqdm import tqdm
import glob

# =================================================================================================================================================================================================
# === Preprocesses the climate2energy-converted data from CESM2 into nc files: for each technology separately, and together (as raw output, and combined with capacity scenarios, as net load). ===
# === Also saves capacity technology as a nc file =================================================================================================================================================
# =================================================================================================================================================================================================

techs = {"PV":"",
             "cooling-demand":"",
             "heating-demand":"_fully-electrified",
             "Wind-power":"",
             "hydro_inflow":"",
             "hydro_ror":"",
            }# all technologies considered

tech_names = ["PV","cooling-demand","heating-demand","Wind_onshore","Wind_offshore","hydro_inflow","hydro_ror"]
out_path = '/net/xenon/climphys/lbloin/energy_boost/'
if __name__ == "__main__":    
    print("Open capacity scenarios")
    installed_capacity = pc.get_installed_capacity(tech_names).to_dataset(name="GWh")
    installed_capacity.to_netcdf(f"{out_path}installed_capacity_scenarios.nc")
    print("Opens Clim2Energy output")
    for scenario in ut.CESM2_REALIZATION_DICT:
        print(scenario)
        outputs = []
        for tech in tqdm(techs):
            if tech == "Wind-power" or tech == "PV":
                # open spatial CFs and save them for available techs
                pc.save_spatial_data(tech,scenario)
            #open and save tech output
            if tech == "Wind-power":
                for onshore in [True, False]:
                    ds_wind = []
                    for turbine in ["E-126_7580","SWT120_3600","SWT142_3150"]: # average over turbine heights
                        ds_wind.append(pc.save_eng_var(scenario,tech,f"_{turbine}_onshore_{onshore}_density_corrected",daily="")[tech])
                    ds_wind = xr.concat(ds_wind,dim="turbine").mean("turbine")
                    ds_wind.to_dataset(name=f"Wind_onshore{onshore}").to_netcdf(f"/net/xenon/climphys/lbloin/energy_boost/country_avgd_Wind-power_{scenario}_onshore{onshore}.nc")
                    outputs.append(ds_wind.to_dataset(name="energy_output"))
            else:
                ds = pc.save_eng_var(scenario,tech, techs[tech],daily="")[tech]
                if tech == "hydro_inflow":
                    ds = ds.resample(time="1h").ffill()/(7*24) # to get hourly values, not weekly
                elif tech == "hydro_ror":
                    ds = ds.resample(time="1h").ffill()/24 # to get hourly values, not daily
                outputs.append(ds.to_dataset(name="energy_output"))
        outputs = xr.concat(outputs,pd.Index(tech_names, name="technology"))
        outputs.to_netcdf(f"{out_path}eng_vars_{scenario}.nc")
        # get absolute output, in terms of capacity
        abs_output = outputs["energy_output"] * installed_capacity.GWh
        abs_output.to_dataset(name="net_load").to_netcdf(f"{out_path}net_load_{scenario}.nc")

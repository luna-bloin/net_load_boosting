import sys
sys.path.append("../utils")
import utils as ut
import preprocess as pc
import hydro_storage as hs
import pandas as pd
import xarray as xr
from tqdm import tqdm
import glob

# =================================================================================================================================================================================================
# === Preprocesses the climate2energy-converted data from CESM2 into nc files: for each technology separately, and together (as raw output, and combined with capacity scenarios, as net load). ===
# === Also saves capacity technology as a nc file =================================================================================================================================================
# =================================================================================================================================================================================================

# === Nomenclature ===
techs = {"PV":"",
         "cooling-demand":"",
         "heating-demand":{"fully_electrified":"_fully-electrified","current_electrified":""},
         "Wind-power":"",
         "hydro_inflow":"",
         "hydro_ror":"",
         "weather-insensitive_demand":"",
        }# all technologies considered
tech_names = ["PV","cooling-demand","heating-demand","Wind_onshore","Wind_offshore","hydro_inflow","hydro_ror"] #naming conventions for data set built here
generation = ["PV","Wind_onshore","Wind_offshore","hydro_ror"] # variables that generate energy
demand = ["heating-demand","cooling-demand","weather-insensitive_demand"] # variables that demand energy

out_path = '/net/xenon/climphys/lbloin/energy_boost/' #save location


if __name__ == "__main__":    
    # === get Clim2Energy converted data sets for historical and SSP370 ===
    print("Opens Clim2Energy output")        
    for scenario in ut.CESM2_REALIZATION_DICT:
        print(scenario)
        outputs = []
        # consider currently electrified and future electrified as separate scenarios
        for heat_scenario in techs["heating-demand"]: 
            out_heat = pc.concat_all_eng_vars(techs,scenario,out_path,heat_scenario)
            outputs.append(xr.concat(out_heat,pd.Index(tech_names, name="technology")))
        outputs = xr.concat(outputs,pd.Index(list(techs["heating-demand"].keys()), name="heating_scenario"))
        # save data set of all converted energy variables for historical and SSP370                    
        outputs.to_netcdf(f"{out_path}eng_vars_{scenario}.nc") # save 

        # === Get the technology capacity scenarios considered ===
        print("Open capacity scenarios")
        installed_capacity = pc.get_installed_capacity(tech_names)
        
        # === Multiply energy variable output by installed capacity to get output in GWh (for all tech scenarios)
        print("Calculate absolute generation")
        abs_output = outputs["energy_output"] * installed_capacity.GWh
        abs_output.to_dataset(name="eng_vars").to_netcdf(f"{out_path}eng_vars_GWh_{scenario}.nc")
        
        # === Calculate simple net load /without/ hydro inflow storage nor transmission effects === 
        print("Calculate simple net load with no extra effects")
        #sum over countries
        abs_vars_country_sum = abs_output.sum("country")
        abs_vars_country_sum.to_dataset(name="eng_vars").to_netcdf(f"{out_path}eng_vars_GWh_country_sum_{scenario}.nc")
        # #sum over technologies
        abs_vars_tech_sum = -abs_output.sel(technology=generation).sum(dim="technology") + abs_output.sel(technology=demand).sum(dim="technology")
        abs_vars_tech_sum.to_dataset(name="net_load").to_netcdf(f"{out_path}net_load_by_country_simple_{scenario}.nc")
        # sum over both 
        abs_vars_tech_sum.to_dataset(name="net_load").sum("country").to_netcdf(f"{out_path}net_load_simple_{scenario}.nc")
    
        # === Calculate simple net load /with/ hydro inflow storage effects === 
        print("Calculate simple net load with hydro storage")
        # calculate hydro_inflow (only keep countries that have hydro inflow)
        hydro_inflow_full = abs_output.sel(technology="hydro_inflow").dropna(dim="country",how="all")
        # open optimized storage from francesco's energy model
        storage_ds = hs.open_storage(scenario)
        storage_roll= storage_ds.rolling(time=24*21,center=True).mean().stack(dim=("member","time")) # rolling average to smooth the curve
        storage_max = storage_ds.max(("member","time")) #max value, to cap the storage
        starting_storage = storage_ds.groupby('time.dayofyear')[1].mean(("member","time")) # mean storage level on January 1st (used as starting point)
        # calculate and save hydro storage effects
        net_load_with_hydro = hs.storage_net_load_all_dims(abs_vars_tech_sum,techs,hydro_inflow_full,storage_roll,storage_max,starting_storage)
        net_load_with_hydro.to_dataset(name="net_load").to_netcdf(f"{out_path}net_load_by_country_hydro_storage_{scenario}.nc")
        net_load_with_hydro.sum("country").to_dataset(name="net_load").to_netcdf(f"{out_path}net_load_hydro_storage_{scenario}.nc")
        
        # === Calculate net load with transmission effects between countries === 
        
        
        
        
        
        
        
        
        
        
        
        
        
        

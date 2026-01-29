import sys
sys.path.append("../utils")
import utils as ut
import preprocess as pc
import hydro_storage as hs
import energy_analysis as ea
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
        }# all technologies considered
tech_names = ["PV","cooling-demand","heating-demand","Wind_onshore","Wind_offshore","hydro_inflow","hydro_ror"] #naming conventions for data set built here
generation = ["PV","Wind_onshore","Wind_offshore","hydro_ror"] # variables that generate energy
demand = ["heating-demand","cooling-demand","weather-insensitive_demand"] # variables that demand energy

out_path = '/net/xenon/climphys/lbloin/energy_boost/' #save location


if __name__ == "__main__":    
    boost = sys.argv[1] # "" if not boost, date of boost otherwise
    try:
        boost_realization = sys.argv[2]
    except:
        boost_realization = "" 
    scenario = sys.argv[3]
    if len(boost) == 0:
        boost_save = ""
    else:
        boost_save = f"_boost_{boost_realization}_{boost}"
        
    # === get Clim2Energy converted data sets for historical and SSP370 ===
    print("Opens Clim2Energy output")        
    outputs = []
    # consider currently electrified and future electrified as separate scenarios
    for heat_scenario in techs["heating-demand"]: 
        out_heat = pc.concat_all_eng_vars(techs,scenario,out_path,heat_scenario,boost,boost_realization)
        outputs.append(xr.concat(out_heat,pd.Index(tech_names, name="technology")))
    outputs = xr.concat(outputs,pd.Index(list(techs["heating-demand"].keys()), name="heating_scenario"))
    # save data set of all converted energy variables for historical and SSP370                    
    outputs.to_netcdf(f"{out_path}eng_vars_{scenario}{boost_save}.nc") # save 

    # === Get the technology capacity scenarios considered ===
    print("Open capacity scenarios")
    installed_capacity = pc.get_installed_capacity(tech_names)
    
    # === Multiply energy variable output by installed capacity to get output in GWh (for all tech scenarios)
    print("Calculate absolute generation")
    abs_output = outputs["energy_output"] * installed_capacity.GWh
    # Add weather-insensitive demand
    if len(boost) == 0:
        members = ["A","B","C"]
    else:
        members = list(range(1,len(outputs.member)+1))
    ds_demand = pc.open_weather_insensitive_demand(scenario,boost,members)
    abs_output = xr.concat([abs_output,ds_demand],dim="technology")
    if len(boost) > 0:
        abs_output = abs_output.sel(time=ut.get_time_plus_delta(boost,64)) # simulations last roughly 60 days
    # Save
    abs_output.to_dataset(name="eng_vars").to_netcdf(f"{out_path}eng_vars_GWh_{scenario}{boost_save}.nc")
    
    # === Calculate simple net load /without/ hydro inflow storage nor transmission effects === 
    print("Calculate simple net load with no extra effects")
    #sum over countries
    abs_vars_country_sum = abs_output.sum("country")
    abs_vars_country_sum.to_dataset(name="eng_vars").to_netcdf(f"{out_path}eng_vars_GWh_country_sum_{scenario}{boost_save}.nc")
    # #sum over technologies
    abs_vars_tech_sum = -abs_output.sel(technology=generation).sum(dim="technology") + abs_output.sel(technology=demand).sum(dim="technology")
    abs_vars_tech_sum.to_dataset(name="net_load").to_netcdf(f"{out_path}net_load_by_country_simple_{scenario}{boost_save}.nc")
    # sum over both 
    abs_vars_tech_sum.to_dataset(name="net_load").sum("country").to_netcdf(f"{out_path}net_load_simple_{scenario}{boost_save}.nc")

    # === Calculate simple net load /with/ hydro inflow storage effects === 
    print("Calculate net load with hydro storage")
    # calculate hydro_inflow (only keep countries that have hydro inflow)
    hydro_inflow_full = abs_output.sel(technology="hydro_inflow").dropna(dim="country",how="all")
    # open optimized storage from francesco's energy model
    storage_ds = hs.open_storage(scenario,boost)
    storage_roll= storage_ds.rolling(time=24*21,center=True).mean().stack(dim=("member","time")) # rolling average to smooth the curve
    storage_max = storage_ds.max(("member","time")) #max value, to cap the storage
    if len(boost) == 0:
        starting_storage = storage_ds.groupby('time.dayofyear')[1].mean(("member","time")) # mean storage level on January 1st (used as starting point)
    else:
        starting_storage = hs.get_boosted_start_storage(boost,storage_ds,boost_realization)
    # calculate and save hydro storage effects
    net_load_with_hydro = hs.storage_net_load_all_dims(abs_vars_tech_sum,techs,hydro_inflow_full,storage_roll,storage_max,starting_storage)
    net_load_with_hydro.to_netcdf(f"{out_path}net_load_by_country_hydro_storage_{scenario}{boost_save}.nc")
    net_load_with_hydro.sum("country").to_netcdf(f"{out_path}net_load_hydro_storage_{scenario}{boost_save}.nc")
    
    # === Calculate net load with transmission effects between countries === 
    print("Calculate transmission effects")
    net_load_transmission = []
    for heating_scenario in net_load_with_hydro.heating_scenario:
        ds_capac = []
        for capacity_scenario in net_load_with_hydro.capacity_scenario:
            ds_mem = []
            for member in net_load_with_hydro.member:
                # create transmission effect calculation framework
                analysis = ea.EnergyAnalysis(net_load_with_hydro.sel(member=member,capacity_scenario=capacity_scenario,heating_scenario=heating_scenario).net_load_adjusted)
                #calculate transmission effects
                ds_mem.append(ea.get_transmission_effect(analysis).to_xarray())
            ds_capac.append(xr.concat(ds_mem,dim=pd.Index(net_load_with_hydro.member,name="member")))
        #do it for all scenarios
        net_load_transmission.append(xr.concat(ds_capac,dim=pd.Index(net_load_with_hydro.capacity_scenario,name="capacity_scenario")))
    net_load_transmission = xr.concat(net_load_transmission,dim=pd.Index(net_load_with_hydro.heating_scenario,name="heating_scenario")).rename({"index":"time"})
    net_load_transmission["time"] = net_load_with_hydro.time # make sure the new data array has the same time dimension
    #save
    net_load_transmission.to_dataset(name="net_load").to_netcdf(f"{out_path}net_load_transmission_{scenario}{boost_save}.nc")

        
        
        
        
        
        
        
        
        
        
        
        
        

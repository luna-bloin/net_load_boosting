import utils as ut
import pandas as pd
import xarray as xr
from tqdm import tqdm
import glob
import numpy as np

# ===============================================
# === open converted data from climate2energy ===
# ===============================================

def read_df_to_xr(file):
    """
    reads pandas df from file and transforms it to xarray
    """
    df = pd.read_csv(file, index_col=0)
    df.columns = pd.to_datetime(df.columns)
    return df.T.to_xarray().to_array(dim="country", name="values").rename({"index":"time"})
    
def save_eng_var(scenario,variable,extra,daily="_daily"):
    """
    Opens the converted Climaet2Energy output for a given variable and scenario.
    :param extra: str used when there are aspects of the variable to open to specify (e.g. heating -> fuly electrified or not?)
    :param daily: str that specifiies whether to resample the data to daily resolution
    """
    save_file = f"/net/xenon/climphys/lbloin/energy_boost/country_avgd_{variable}_{scenario}{extra}{daily}.nc"
    try:
        return xr.open_dataset(glob.glob(save_file[0]))
    except:
        members = list(ut.CESM2_REALIZATION_DICT[scenario].keys())
        ds_all = []
        for mem in members:
            path = f"/net/xenon/climphys/lbloin/CESM2energy/output/bias_correction/{mem}/{scenario}/{mem}/output_variables/"
            if extra == "":
                files = sorted(glob.glob(f"{path}{variable}_[0-9][0-9][0-9][0-9].csv"))
            else:
                files = sorted(glob.glob(f"{path}{variable}_*{extra}.csv"))
            dss = []
            for file in files:
                dss.append(read_df_to_xr(file))
            ds_all.append(xr.concat(dss,dim="time"))
        if scenario == "SSP245":
            ds_all = ds_all[0] #only one member
        else:
            ds_all = xr.concat(ds_all,dim="member")
            ds_all["member"] = members
        ds_all = ds_all.to_dataset(name=variable).convert_calendar("noleap")
        if daily == "_daily":
            ds_all = ds_all.resample(time="1D").sum()
        ds_all.to_netcdf(save_file)
        return ds_all.load()

def save_spatial_data(variable,scenario):
    save_file = f"/net/xenon/climphys/lbloin/energy_boost/CF_{variable}_{scenario}.nc"
    if glob.glob(save_file) == []:
        members = list(ut.CESM2_REALIZATION_DICT[scenario].keys())
        ds_all = []
        for mem in members:
            path = f"/net/xenon/climphys/lbloin/CESM2energy/output/bias_correction/{mem}/{scenario}/{mem}/output_variables/"
            files = sorted(glob.glob(f"{path}{variable}*.nc"))
            ds_all.append(xr.open_mfdataset(files))
        ds_all = xr.concat(ds_all,dim="member")
        ds_all.to_netcdf(save_file)
    return None

def concat_all_eng_vars(techs,scenario,out_path,heat_scenario):
    eng_vars = []
    # open data for all technologies considered
    for tech in tqdm(techs):
        print(tech)
        if tech == "Wind-power" or tech == "PV":
            # open spatial CFs and save them for available techs
            save_spatial_data(tech,scenario)
        # open and save tech output from climate2energy
        if tech == "Wind-power":
            for onshore in [True, False]:
                ds_wind = []
                for turbine in ["E-126_7580","SWT120_3600","SWT142_3150"]: # average over turbine heights
                    ds_wind.append(save_eng_var(scenario,tech,f"_{turbine}_onshore_{onshore}_density_corrected",daily="")[tech])
                ds_wind = xr.concat(ds_wind,dim="turbine").mean("turbine")
                ds_wind.to_dataset(name=f"Wind_onshore{onshore}").to_netcdf(f"{out_path}country_avgd_Wind-power_{scenario}_onshore{onshore}.nc")
                eng_vars.append(ds_wind.to_dataset(name="energy_output"))
        elif tech == "heating-demand":
            ds = save_eng_var(scenario,tech, techs[tech][heat_scenario],daily="")[tech] 
            eng_vars.append(ds.to_dataset(name="energy_output"))
        else:
            ds = save_eng_var(scenario,tech, techs[tech],daily="")[tech]
            if tech == "hydro_inflow":
                ds = ds.resample(time="1h").ffill()/(7*24) # to get hourly values, not weekly
            elif tech == "hydro_ror":
                ds = ds.resample(time="1h").ffill()/24 # to get hourly values, not daily
            eng_vars.append(ds.to_dataset(name="energy_output"))
    return eng_vars

def preproc_cesm_z500(scenario,member):
    """
    opens the non bias corrected daily Z500 data for a given scenario and member
    """
    realization = ut.CESM2_REALIZATION_DICT[scenario][member]
    # historical is called HIST in the file system
    if scenario == "historical":
        scen_file = "HIST"
    else:
        scen_file = scenario
    # open and concat all years in time range
    z500s = []
    for year in tqdm(ut.get_time_range(scenario)):
        file = f"/net/meso/climphys/cesm212/b.e212.B{scen_file}cmip6.f09_g17.{realization}/archive/atm/hist/b.e212.B{scen_file}cmip6.f09_g17.{realization}.cam.h1.{year}-01-01-00000.nc"
        z500s.append(ut.select_Europe(ut.zero_mean_longitudes(xr.open_dataset(file))))
    return xr.concat(z500s,dim="time")

# ===============================
# === Read capacity scenarios ===
# ===============================

def read_future_capacity(country,var,cap_type="future_high"):
    """
    Reads the future capacity scenario of a given variable and country, according to the ENTSO-E TYNDP future scenario
    :param cap_type: whether to choose the high or low future scenario
    :param var: can only be solar, wind offshore or wind onshore
    """
    var_to_sheet_nb = {"Solar":"61","Wind Onshore":"62","Wind Offshore": "63"}
    sheet_name = var_to_sheet_nb[var]
    df=pd.read_excel("../../inputs/future_capacity.xlsx",sheet_name=sheet_name,index_col=1,skiprows=[0,1])
    df =df[0:-3] # to avoid including total numbers
    if "high" in cap_type:
        return df[df.index.astype(str).str.contains(country)]["2050.1"].sum()/1000
    elif "low" in cap_type:
        return df[df.index.astype(str).str.contains(country)][2050].sum()/1000

def read_current_capacity(country,var):
    """
    reads the current technology capacity, according to ENTSO-E power stats for a given country and var
    """
    df = pd.read_csv("../../inputs/net_generation_capacity_2024.csv",delimiter='\t',index_col=0)
    df_country = df[df["Country"] == country] # find data for selected country
    obs = df_country[df_country["Category"] == var]["ProvidedValue"]
    if len(obs) > 0:
        return obs.item()/1000
    else:
        return 0

def get_obs(country,capacity_type,var):
    """
    read the right capacity scenario for a given var, country"""
    if capacity_type =="current":
        return read_current_capacity(country,var)
    if capacity_type == "future":
        return read_future_capacity(country,var)

def get_avg_CFs(country):
    """
    Get the average capacity factor for PV and on- and offshore wind. We do a mean over the historical and future period, although differences are in the order of max 5%
    :param country: str of the country to get the CF for
    """
    CFs = []
    for var in ["PV", "Wind True", "Wind False"]:
        CF = []
        for scenario in ["historical","SSP370"]: #avearge over the two scenarios (differences are very small)
            if var == "PV":
                ds = xr.open_dataset(f"/net/xenon/climphys/lbloin/energy_boost/country_avgd_PV_{scenario}.nc").PV
            else:
                onshore = var[5:]
                ds = xr.open_dataset(f"/net/xenon/climphys/lbloin/energy_boost/country_avgd_Wind-power_{scenario}_onshore{onshore}.nc")[f"Wind_onshore{onshore}"]
            if country in ds.country.values:
                CF.append(ds.sel(country=country).mean(("member","time")).item())
            else:
                CF.append(0)
        CFs.append(np.array(CF).mean())
    return CFs

def get_syn_capac_scenario(base_scenario,change_factor,country):
    """
    Create synthetic technology capacity scenarios, that change the factor of generated energy between wind and pv, but not the total energy production.
    Additionally, we constrain the system so that the ratio of onshore to offshore stays the same.
    This creates a system of linear equations, where the three unknowns are the new installed capacities for pv,wind onshore and wind offshore (IC2_pv, IC2_won, IC2_wof):
    (1) IC2_pv*CF_pv + IC2_won*CF_won + IC2_wof*CF_wof = IC1_pv*CF_pv + IC1_won*CF_won + IC1_wof*CF_wof
    (2) change_factor * IC2_pv = IC2_won, IC2_wof
    (3) IC1_wof / IC1_won = IC2_wof / IC2_won
    :param base_scenario:
    :param change_factor: 
    :param country: country for which the capacities come from
    """
    # get the installed capacity of the base scenario for pv,wind onshore and wind offshore
    IC1_pv, IC1_won, IC1_wof = [base_scenario[i] for i in [0, 3, 4]]
    if IC1_pv == IC1_won == IC1_wof == 0:
        IC2_pv, IC2_won, IC2_wof = 0,0,0
    else:
        # get the average capacity factors for pv,wind onshore and wind offshore
        CF_pv, CF_won, CF_wof = get_avg_CFs(country)
        # total energy output of base scenario
        tot_out = IC1_pv*CF_pv + IC1_won*CF_won + IC1_wof*CF_wof
        # define the system of linear equations and solve for IC2_pv, IC2_won, IC2_wof
        left_side = np.array([[CF_pv, CF_won,CF_wof],[change_factor,-1,-1],[0,IC1_wof/IC1_won,-1]])
        right_side = np.array([tot_out,0,0])
        IC2_pv, IC2_won, IC2_wof = np.linalg.inv(left_side).dot(right_side)
    return IC2_pv, IC2_won, IC2_wof

def get_installed_capacity(techs):
    """
    Create a data set of all considered capacity scenarios for the study
    """
    file = f"/net/xenon/climphys/lbloin/energy_boost/installed_capacity_scenarios.nc"
    try:
        return xr.open_dataset(glob.glob(file)[0])
    except:
        
        countries = list(ut.country_name_to_country_code(None).keys())
        country_da = []
        for country in countries:
            scen = ["current","future"]
            synthetic_scenarios = ["future_wind_x2","future_wind_x0.5"]
            print(country)
            da = []
            for capacity_type in scen:
                code = ut.country_name_to_country_code(country)
                # open capacities for wind and solar
                wind_onshore = get_obs(code,capacity_type,"Wind Onshore")
                wind_offshore = get_obs(code,capacity_type,"Wind Offshore") #combine off and onshore
                pv = get_obs(code,capacity_type,"Solar")
                da.append([pv,1,1,wind_onshore,wind_offshore,1,1]) #hydro and demandtechs do not need capacity adjustment, therefore they are just noted as 1 here
            # add the two synthetically created scenarios
            for syn_scen in synthetic_scenarios:
                change_factor = float(syn_scen[13:])
                pv,wind_onshore,wind_offshore = get_syn_capac_scenario(da[1],change_factor,country)
                da.append([pv,1,1,wind_onshore,wind_offshore,1,1])
                scen.append(syn_scen)
            country_da.append(da)
            # create data array
        installed_capacity = xr.DataArray(
            data=country_da,
            dims=["country","capacity_scenario","technology"],
            coords=dict(
                country=countries,
                technology=techs,
                capacity_scenario=scen
            ),
        ).to_dataset(name="GWh")
        installed_capacity.to_netcdf(file)
        return installed_capacity
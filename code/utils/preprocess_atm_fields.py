import utils as ut
import xarray as xr
import numpy as np
from tqdm import tqdm
import glob

# ============================================
# === Functions inspired by Climate2energy ===
# ============================================

def find_height(ds):
    """
    Calculates the height of dataset model levels
    :param ds:
    :return: ds: - same dataset as input, with added data array ds["height"]
    """
    if "Z3" not in ds.data_vars:
        raise ValueError(
            "Error: dataset does not have variable 'Z3', necessary for height calculation."
        )
    ds_orog = xr.open_dataset(
        "/net/meso/climphys/cesm212/inputfiles/BSSP370cmip6/atm/cam/topo/fv_0.9x1.25_nc3000_Nsw042_Nrs008_Co060_Fi001_ZR_sgh30_24km_GRNL_c170103.nc"
    )
    ds_orog = ut.select_Europe(ut.zero_mean_longitudes(ds_orog))
    ds_orog = (
        ds_orog["PHIS"] / 9.80665
    )  # geopotential reported in m**2/s**2 and divided by earth acceleration according to https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html

    ds["height"] = ds["Z3"] - ds_orog
    return ds

def temp_cel(ds):
    """
    returns the temperature dataset ds in celsius
    """
    if "temperature" in ds.data_vars:
        ds["temperature"] = ds["temperature"] - 273.15
        ds["temperature"].attrs["units"] = "degrees C"
        return ds
    else:
        "temperature is not in this dataset"
        return None

def interpolate_wind_xr(ds, output_height=120):
    """
    Calculates wind speeds at output height using data at evolving heights and
    the power law. Execution is done for one timestep here.

    The power law exponent is fitted per time step and location, thereby accounting
    for the fact that wind profiles do not always look the same

    It is assumed that the Dataset ds only contains two levels that are close to the
    hub height.

    This function is taken and modified from lucas_3dwinds
    """
    vertical_dim = "lev"
    # Identify index of upper and lower level to be used here
    height_max = 0
    height_min = 1
    # Calculate power law exponent alpha
    alpha = np.log(
        ds["S"].isel(lev=height_max) / ds["S"].isel(lev=height_min)
    ) / np.log(ds["height"].isel(lev=height_max) / ds["height"].isel(lev=height_min))
    # Interpolate to output height
    y_hub = (
        ds["S"].isel(lev=height_min)
        * (output_height / ds["height"].isel(lev=height_min)) ** alpha
    )
    ds_hub = y_hub.to_dataset(name="s_hub")
    ds_hub["s_hub"].attrs = {"long_name": "Wind speed at hub height [m/s]"}
    ds_hub = ds_hub.drop(vertical_dim)
    return ds_hub, alpha

# === Open atmospheric variables before bias correction ===

def preproc_atm_vars(scenario,member,path,boost_date=""):
    """
    Open the wind, radiation, temperature and z500 for a given scenario and member
    """
    # find input file parameters
    realization = ut.CESM2_REALIZATION_DICT[scenario][member]
    # historical is called HIST in the file system
    if scenario == "historical":
        scen_file = "HIST"
    else:
        scen_file = scenario
    if len(boost_date)==0:
        # open the atmospheric dataset
        ds_atm = []
        for year in tqdm(ut.get_time_range(scenario)):
            file = f"/net/meso/climphys/cesm212/b.e212.B{scen_file}cmip6.f09_g17.{realization}/archive/atm/hist/b.e212.B{scen_file}cmip6.f09_g17.{realization}.cam.h6.{year}-01-01-03600.nc"
            ds_atm.append(ut.select_Europe(ut.zero_mean_longitudes(xr.open_dataset(file).isel(lev=slice(29, 32), ilev=slice(29, 33)))).resample(time="1D").mean()) # daily data, to not take up too much space
        ds_atm = xr.concat(ds_atm,dim="time")
    else:
        def preproc_boost(ds):
            return ut.select_Europe(ut.zero_mean_longitudes(ds.isel(lev=slice(29, 32), ilev=slice(29, 33)))).resample(time="1D").mean()
        files = sorted(glob.glob(f"/net/meso/climphys/cesm212/boosting/archive/B{scen_file}cmip6.100{realization}.{boost_date}.ens*/atm/hist/B{scen_file}cmip6.100{realization}.{boost_date}.ens*.cam.h6.*.nc"))
        ds_atm = xr.open_mfdataset(file,preprocess=preproc_boost,concat_dim="member", combine="nested")
        ds_atm["member"] = list(range(1,len(ds_atm.member)+1))
    # Wind
    ds_wind = ds_atm.isel(lev=slice(-2, None))  # lowermost 2 levels
    ds_wind = np.sqrt(ds_wind["U"] ** 2 + ds_wind["V"] ** 2)
    ds_wind = ds_wind.to_dataset(name="S")  # call winds S here because they are still at model level
    ds_wind["Z3"] = ds_atm["Z3"]
    ds_wind = find_height(ds_wind)
    # radiation
    ds_PV = ds_atm["FSDS"].to_dataset(name="global_horizontal")
    # temperature
    ds_temp = ds_atm["TREFHT"].to_dataset(name="temperature")
    ds_temp = temp_cel(ds_temp)  # temperature in celsius    

    if len(boost_date)==0:
        # Z500
        z500s = []
        for year in tqdm(ut.get_time_range(scenario)):
            file = f"/net/meso/climphys/cesm212/b.e212.B{scen_file}cmip6.f09_g17.{realization}/archive/atm/hist/b.e212.B{scen_file}cmip6.f09_g17.{realization}.cam.h1.{year}-01-01-00000.nc"
            z500s.append(ut.select_Europe(ut.zero_mean_longitudes(xr.open_dataset(file))))
        ds_Z500 = xr.concat(z500s,dim="time")
        # save
        ds_wind.to_netcdf(f"{path}Raw_CESM2_s100_{scenario}_{member}.nc")
        ds_PV.to_netcdf(f"{path}Raw_CESM2_global-horizontal_{scenario}_{member}.nc")
        ds_temp.to_netcdf(f"{path}Raw_CESM2_temperature_{scenario}_{member}.nc")
        ds_Z500.to_netcdf(f"{path}Raw_CESM2_Z500_{scenario}_{member}.nc")
    else:
        # Z500
        def preproc(ds):
            return ut.select_Europe(ut.zero_mean_longitudes(ds))
        files = sorted(glob.glob(f"/net/meso/climphys/cesm212/boosting/archive/B{scen_file}cmip6.100{realization}.{boost_date}.ens*/atm/hist/B{scen_file}cmip6.100{realization}.{boost_date}.ens*.cam.h1.*.nc"))
        z500 = xr.open_mfdataset(files,preprocess=preproc,concat_dim="member", combine="nested")
        z500["member"] = list(range(1,len(z500.member)+1))
        # save
        ds_wind.to_netcdf(f"{path}Raw_CESM2_s100_{scenario}_boost_{member}_{boost_date}.nc")
        ds_PV.to_netcdf(f"{path}Raw_CESM2_global-horizontal_{scenario}_boost_{member}_{boost_date}.nc")
        ds_temp.to_netcdf(f"{path}Raw_CESM2_temperature_{scenario}_boost_{member}_{boost_date}.nc")
        ds_Z500.to_netcdf(f"{path}Raw_CESM2_Z500_{scenario}_boost_{member}_{boost_date}.nc")
    return None

def preproc_cesm2(scenario,member,path,var):
    file = f"{path}Raw_CESM2_{var}_{scenario}_{member}.nc"
    if len(glob.glob(file)) ==0:
        preproc_atm_vars(scenario,member,path)
    return xr.open_dataset(glob.glob(file)[0])

def preproc_cesm2_boosted(boost_date,scenario,parent_member,path,var):
    file = f"{path}Raw_CESM2_{var}_{scenario}_boost_{parent_member}_{boost_date}.nc"
    if len(glob.glob(file)) ==0:
        preproc_atm_vars(scenario,member,path)
    return xr.open_dataset(glob.glob(file)[0])

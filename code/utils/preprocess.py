import utils as ut
import pandas as pd
import xarray as xr
from tqdm import tqdm
import glob

# === open converted data from climate2energy ===

def read_df_to_xr(file):
    """
    reads pandas df from file and transforms it to xarray
    """
    df = pd.read_csv(file, index_col=0)
    df.columns = pd.to_datetime(df.columns)
    return df.T.to_xarray().to_array(dim="country", name="values").rename({"index":"time"})
    
def save_eng_var(scenario,variable,extra,daily="_daily"):
    save_path = f"/net/xenon/climphys/lbloin/energy_boost/country_avgd_{variable}_{scenario}{extra}{daily}.nc"
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
    ds_all = ds_all.to_dataset(name=variable)
    if daily == "_daily":
        ds_all = ds_all.resample(time="1D").sum()
    ds_all.to_netcdf(save_path)
    return ds_all.load()

def save_spatial_data(variable,scenario):
    save_path = "/net/xenon/climphys/lbloin/energy_boost/"
    members = list(ut.CESM2_REALIZATION_DICT[scenario].keys())
    ds_all = []
    for mem in members:
        path = f"/net/xenon/climphys/lbloin/CESM2energy/output/bias_correction/{mem}/{scenario}/{mem}/output_variables/"
        files = sorted(glob.glob(f"{path}{variable}*.nc"))
        ds_all.append(xr.open_mfdataset(files))
    ds_all = xr.concat(ds_all,dim="member")
    ds_all.to_netcdf(f"{save_path}CF_{variable}_{scenario}.nc")
    return None

def open_bced_clim_data(scenario):
    """
    opens bias corrected climate data for several members and scenarios.
    :param var: (str or list) variable of climate data to open. If it is a list, it opens data of all variables and concatenates it into one ds
    returns loaded xr dataset of data
    """
    members = list(ut.CESM2_REALIZATION_DICT[scenario].keys())
    vars_to_open = ["global-horizontal", "s100","temperature","discharge"]
    path = f"/net/xenon/climphys/lbloin/energy_boost/clim_data_{scenario}_bced.nc"
    file = glob.glob(path)
    if file != []:
        return xr.open_dataset(file[0])
    else:
        ds_all = []
        for var in vars_to_open:
            ds_climate = []
            for mem in tqdm(members):
                files = sorted(glob.glob(f"/net/xenon/climphys/lbloin/CESM2energy/output/bias_correction/{mem}/{scenario}/{mem}/atmospheric_variables/bced_CESM2_{var}_*.nc"))
                ds = xr.open_mfdataset(files).convert_calendar("noleap")
                ds_climate.append(ds)
            if scenario == "SSP245":
                ds_climate = ds_climate[0] #only one member
            else:
                ds_climate = xr.concat(ds_climate, dim = "member")
            ds_climate["member"] = members
            if var != "discharge":
                ds_climate = ds_climate.resample(time="1D").mean()
            ds_climate.to_netcdf(path)
            ds_all.append(ds_climate)
        ds_all = xr.concat(ds_all, pd.Index(vars_to_open, name="variable"))
        return ds_daily

# === Read capacity scenarios ===

def read_future_capacity(country,var,cap_type):
    var_to_sheet_nb = {"Solar":"61","Wind Onshore":"62","Wind Offshore": "63"}
    sheet_name = var_to_sheet_nb[var]
    df=pd.read_excel("../../inputs/future_capacity.xlsx",sheet_name=sheet_name,index_col=1,skiprows=[0,1])
    df =df[0:-3] # to avoid including total numbers
    if "high" in cap_type:
        return df[df.index.astype(str).str.contains(country)]["2050.1"].sum()/1000
    elif "low" in cap_type:
        return df[df.index.astype(str).str.contains(country)][2050].sum()/1000

def read_current_capacity(country,var):
    df = pd.read_csv("../../inputs/net_generation_capacity_2024.csv",delimiter='\t',index_col=0)
    df_country = df[df["Country"] == country] # find data for selected country
    obs = df_country[df_country["Category"] == var]["ProvidedValue"]
    if len(obs) > 0:
        return obs.item()/1000
    else:
        return 0

def get_obs(country,capacity_type,var):
    if capacity_type =="current":
        return read_current_capacity(country,var)
    elif "future" in capacity_type:
        return read_future_capacity(country,var,capacity_type)


def get_installed_capacity(techs):
    countries = list(ut.country_name_to_country_code(None).keys())
    scen = ["current","future_high","future_low"]
    capac_scen = []
    for capacity_type in scen:
        da = []
        for country in countries:
            code = ut.country_name_to_country_code(country)
            # open capacities for wind and solar
            wind_onshore = get_obs(code,capacity_type,"Wind Onshore")
            wind_offshore = get_obs(code,capacity_type,"Wind Offshore") #combine off and onshore
            pv = get_obs(code,capacity_type,"Solar")
            da.append([pv,1,1,wind_onshore,wind_offshore,1,1]) #hydro and demands do not need capacity adjustment, therefore they are just noted as 1 here
        capac_scen.append(da)
        # create data array
    installed_capacity = xr.DataArray(
        data=capac_scen,
        dims=["capacity_scenario","country","technology"],
        coords=dict(
            country=countries,
            technology=techs,
            capacity_scenario=scen
        ),
    )
    return installed_capacity
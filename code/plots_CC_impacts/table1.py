import xarray as xr

installed_capacity_wind_solar = xr.open_dataset(f"/net/xenon/climphys/lbloin/energy_boost/installed_capacity_scenarios.nc").GWh.sel(technology=["PV","Wind_onshore","Wind_offshore"],capacity_scenario=["future","future_wind_x2","future_wind_x0.5"])
mn_installed_capacity_wind_solar = installed_capacity_wind_solar.mean("country")
relative_installed_capacity = mn_installed_capacity_wind_solar/mn_installed_capacity_wind_solar.sum("technology")

for scenario in relative_installed_capacity.capacity_scenario.values:
    print(scenario)
    for tech in relative_installed_capacity.technology.values:
        print(tech, f"{relative_installed_capacity.sel(capacity_scenario=scenario,technology=tech).values:.2}")
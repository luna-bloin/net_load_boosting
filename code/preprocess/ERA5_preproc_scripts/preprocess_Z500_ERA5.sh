output_path=/net/xenon/climphys/lbloin/energy_boost/

for year in {1995..2014}
    do
    # The following cdo comment is split into multiple lines to increase legibility. Here is what happens per line
    # 1) change name and divide by 9.81 to get height in m
    # 2) select 500 hPa level;
    # 3) Choose European region of interest
    # 4) Remap bilinearly to CESM2 grid
    # 5) From hourly to daily values
    # 6) Choose region with 5 degrees sponge in all directions (just to speed up computation)
    # 7) ERA5 input file
    # 8) output file
    cdo -b F32 -setattribute,Z500@units="m" -chname,z,Z500 -divc,9.81 \
    -sellevel,500.0 \
    -sellonlatbox,-25,35,30,75 \
    -remapbil,../../inputs/CESM_atm_grid.txt \
    -daymean \
    -sellonlatbox,-30,40,25,80 \
    /net/atmos/data/ERA5_deterministic/recent/0.25deg_lat-lon_1h/processed/regrid/era5_deterministic_recent.z.025deg.1h.${year}.nc \
    ${output_path}/tmp_${year}_z500_ERA5.nc
done
# merge all years
cdo mergetime ${output_path}/tmp_*_z500_ERA5.nc ${output_path}/Raw_ERA5_z500.nc
# clean up
rm ${output_path}/tmp_*_z500_ERA5.nc
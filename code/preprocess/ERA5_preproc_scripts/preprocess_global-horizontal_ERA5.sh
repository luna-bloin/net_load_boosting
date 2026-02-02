output_path=/net/xenon/climphys/lbloin/energy_boost/
input_path_ERA5=/net/atmos/data/ERA5_deterministic/recent/0.25deg_lat-lon_1h/processed/regrid/

for year in {1995..2014}
    do
    # The following cdo comment is split into multiple lines to increase legibility. Here is what happens per line
    # 1) Unit conversion to W/m2
    # 2) rename from ssrd to global_horizontal
    # 3) Choose European region of interest
    # 4) Remap conservatively to CESM2 grid
    # 5) Do a mean over the day in terms of values
    # 6) Choose region with 5 degrees sponge in all directions (just to speed up computation)
    # 7) ERA5 input file
    # 8) output file
    cdo -b F64 -divc,3600 -setattribute,global_horizontal@units="W/m^2" \
    -chname,ssrd,global_horizontal \
    -sellonlatbox,-25,35,30,75 \
    -remapcon,../../inputs/CESM_atm_grid.txt \
    -daymean \
    -sellonlatbox,-30,40,25,80 \
    ${input_path_ERA5}/era5_deterministic_recent.ssrd.025deg.1h.${year}.nc \
    ${output_path}/tmp_${year}_global_horizontal_ERA5.nc
done
# merge all years
cdo mergetime ${output_path}/tmp_*_global_horizontal_ERA5.nc ${output_path}/Raw_ERA5_global-horizontal.nc
# clean up
rm ${output_path}/tmp_*_global_horizontal_ERA5.nc
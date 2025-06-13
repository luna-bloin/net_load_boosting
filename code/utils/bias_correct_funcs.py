import pandas as pd
import numpy as np
import xarray as xr
from bias_correction import BiasCorrection

def bias_correct_per_loc(reference, model, da, method="basic_quantile"):
    """
    Computes bias correction using the model output during historical period (model)
    and the ground truth (reference) and applies  the correction to future model output (da).

    reference, model and da have to be xr.DataArrays at a single location
    """
    if np.isnan(da).all():
        return da
    else:
        bc = BiasCorrection(pd.Series(reference), pd.Series(model), pd.Series(da))
        return bc.correct(method=method)


def bias_correct_xarray(da, ref_da, hist_da, method="basic_quantile"):
    """
    bias correction applied to use for xarray data arrays.

    """
    corrected = xr.apply_ufunc(
        bias_correct_per_loc,
        ref_da,
        hist_da,
        da,
        vectorize=True,
        input_core_dims=[["time"], ["time"], ["time"]],
        exclude_dims=set(("time",)),
        output_core_dims=[["time"]],
        kwargs={"method": method},
    )
    corrected["time"] = da["time"]  # to restore time coordinate in dataarray
    return corrected.squeeze()

def bias_correct_dataset(to_bc, cesm_hist, era5_hist, method="basic_quantile"):
    """
    takes a data array da of a chosen variable var, and returns the
    bias corrected version (using package bias_correction).
    """
    # the reference dataset has slightly different values for the dimension "lat" (max 10E-14) due to different segmentation in cdo/python. this fixes it
    era5_hist["lat"] = cesm_hist.lat
    # bias_correction
    corrected = bias_correct_xarray(to_bc.load(), era5_hist.load(), cesm_hist.load())
    return corrected.squeeze()
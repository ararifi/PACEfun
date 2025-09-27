# %%
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import earthaccess
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
import sys
sys.path.append("/home/jovyan/PACEfun")
import helper
from dask.distributed import Client

# %%
client = Client()
auth = earthaccess.login("login.netrc")

# %%
# --- TIME ---

tspan = ("2024-09-01 00:00", "2025-10-01 00:00")

# --- REGION ---

# 42.69174831596751, 21.058555012396997
region = (41.7, 20.36, 43.7, 22.06)
mid = (21.058555012396997, 42.69174831596751)
ext = 1
region = (mid[0] - ext , mid[1] - ext, mid[0] + ext, mid[1] + ext)
region

# %%
results_spx = earthaccess.search_data(
    short_name="PACE_SPEXONE_L2_AER_RTAPLAND",
    cloud_hosted=True,
    bounding_box=region,
    temporal=tspan,
)
paths_spx = earthaccess.open(results_spx)

# %%
results_oci = earthaccess.search_data(
    short_name="PACE_OCI_L2_AER_UAA",
    cloud_hosted=True,
    bounding_box=region,
    temporal=tspan,
)
paths_oci = earthaccess.open(results_oci)

# %%
wv = 550
wv_idx = helper.get_wv_idx(paths_spx[0], wv)
crs, shape_tmp, transform_tmp = helper.crs_template(paths_spx[0], "aot", wv)
# Select best data set who covers the most region for the study of interest
shape, transform, _ = helper.grid_aligned_subset(region, transform_tmp, shape_tmp)

# %%

kwargs = {"combine": "nested", "concat_dim": "time"}
attrs = xr.open_mfdataset(paths_spx, preprocess=helper.time_from_attr, **kwargs)
futures = client.map(
    helper.grid_match,
    paths_spx,
    dst_crs=crs,
    dst_shape=shape,
    dst_transform=transform,
    var="aot",
    wv_idx=wv_idx
)
da_spx = xr.combine_nested(client.gather(futures), concat_dim="time")
da_spx["time"] = attrs["time"]

# %%
wv = 550
wv_idx = 550
crs, shape_tmp, transform_tmp = helper.crs_template(paths_oci[0], "Aerosol_Optical_Depth", wv)
# Select best data set who covers the most region for the study of interest
shape, transform, _ = helper.grid_aligned_subset(region, transform_tmp, shape_tmp)

# %%

kwargs = {"combine": "nested", "concat_dim": "time"}
attrs = xr.open_mfdataset(paths_oci, preprocess=helper.time_from_attr, **kwargs)
futures = client.map(
    helper.grid_match,
    paths_oci,
    dst_crs=crs,
    dst_shape=shape,
    dst_transform=transform,
    var="Aerosol_Optical_Depth",
    wv_idx=wv_idx
)
da_oci = xr.combine_nested(client.gather(futures), concat_dim="time")
da_oci["time"] = attrs["time"]

# %%
#da_oci_matched = da_oci.reindex(
#    time=da_spx.time,
#    method="nearest",
#    tolerance="3H"
#)

da_oci_time_matched = da_oci.interp(time=da_spx["time"], method="nearest")
da_oci_matched = da_oci_time_matched.interp(
    longitude=da_spx["longitude"],
    latitude=da_spx["latitude"],
    method="linear"
)


# %%
nc_path='/home/jovyan/nc_files'
da_spx.to_netcdf(f"{nc_path}/spx_aod.nc")
da_oci.to_netcdf(f"{nc_path}/oci_aod.nc")
da_oci_matched.to_netcdf(f"{nc_path}/oci_spx_aod.nc")

da_oci_time_matched.to_netcdf(f"{nc_path}/oci_spxTime_aod.nc")

# %%
cross_lonlat = (21.058555012396997, 42.69174831596751)

# Static mean panel comparison using helper
helper.plot_mean_panels(
    [da_spx, da_oci_matched],
    region=region,
    titles=["SPX Mean", "OCI Mean"],
    crosshair=cross_lonlat,
    cmap="viridis",
    vmin=0.0,
    vmax=0.2,
    background="satellite",
    tiles_zoom=12,
)

# %%
# Time animation using helper
_, _, ani = helper.animate_panels(
    [da_spx, da_oci_time_matched],
    region=region,
    titles=["SPX", "OCI Time-Matched"],
    crosshair=cross_lonlat,
    cmap="viridis",
    vmin=0.0,
    vmax=0.2,
    interval=600,
    save_path="PACEfun/movies/spx_oci_comparison.gif",
    writer="pillow",
    dpi=120,
)


# %%
# Interactive time slider (Jupyter) using helper
helper.interactive_panels(
    [da_spx, da_oci_time_matched],
    region=region,
    titles=["SPX", "OCI Time-Matched"],
    crosshair=cross_lonlat,
    vmin=0.0,
    vmax=0.2,
)




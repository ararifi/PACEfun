# %%

# seq 0 50 | parallel -j 8 python chla_verification_oci.py {}


import earthaccess
fs = earthaccess.get_fsspec_https_session()

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
import os
import sys
from pathlib import Path
sys.path.append("/work")
from helper import *

# %%
data_dir = Path("/work/chla_NAO/")
interp_dir = Path("/work/chla_NAO/")

# NOA CHLA
region = (-88.242188, 4.214943, -2.988281, 44.465151)
tspan = ("2024-09-01 00:00", "2024-11-01 00:00")

# PATHS
paths_oci_all = sorted(data_dir.glob("oci/*"))

# Projection/grid template
crs, shape_tmp, transform_tmp = crs_template(paths_oci_all[0], "chlor_a")
shape, transform, _ = grid_aligned_subset(region, transform_tmp, shape_tmp)

# %%
# Batch handling
batch_size = 10

# Take command-line argument for batch index
# Usage: python process_oci_batch.py <batch_index>
if len(sys.argv) < 2:
    print("Usage: python process_oci_batch.py <batch_index>")
    sys.exit(1)

batch = int(sys.argv[1])

start_ind = batch * batch_size
end_ind = min((batch + 1) * batch_size, len(paths_oci_all))

if start_ind >= len(paths_oci_all):
    print(f"Batch {batch} out of range, nothing to process.")
    sys.exit(0)

paths_oci = paths_oci_all[start_ind:end_ind]
output_path = interp_dir / f"oci_L3M/oci_{batch_size}_{batch}.nc"

# Skip if file already exists
if output_path.exists():
    print(f"Skipping batch {batch}, file already exists: {output_path}")
    sys.exit(0)

print(f"Processing batch {batch}: indices {start_ind}-{end_ind}, output -> {output_path}")

# %%
kwargs = {"combine": "nested", "concat_dim": "time"}
attrs = xr.open_mfdataset(paths_oci, preprocess=time_from_attr, **kwargs)

results = []
for path in paths_oci:
    result = grid_match(
        path,
        dst_crs=crs,
        dst_shape=shape,
        dst_transform=transform,
        var="chlor_a",
    )
    results.append(result)

da_oci = xr.combine_nested(results, concat_dim="time")
da_oci["time"] = attrs["time"]

# %%
da_oci.to_netcdf(output_path)
print(f"✅ Saved batch {batch} to {output_path}")
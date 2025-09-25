# %%


from pathlib import Path

import cartopy.crs as ccrs
import earthaccess
import matplotlib.pyplot as plt
import numpy as np
import requests, pdb
import xarray as xr

auth = earthaccess.login(persist=True)
fs = earthaccess.get_fsspec_https_session()

tspan = ("2024-04-01 00:00", "2025-10-01 00:00")
auth = earthaccess.login("../login.netrc")
mid = (21.058555012396997, 42.69174831596751)
ext = 1
bbox = (mid[0] - ext , mid[1] - ext, mid[0] + ext, mid[1] + ext)
clouds = (0, 50) #cloud cover 0-50%

#===============================================
#1. get lists of timestamps that sweeps thru our coordinates

#2. make a for loop for all the timestamps and download oci aod (because oci aod is not offered in earthaccess, we cannot designate bbox)
#===============================================
results = earthaccess.search_data(
    short_name="PACE_OCI_L2_AER_UAA", #example var.
    temporal=tspan,
    bounding_box=bbox,
    cloud_cover=clouds,
)
print(results)
print(len(results))
# %%
tstamps = [] #lists of timestamps
for i in range(len(results)):
    fname = results[i]['meta']['native-id']
    rec = fname.strip().split('.')
    tstamps.append(rec[1])
    
# %%
OB_DAAC_PROVISIONAL = "https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/"
for tstamp in tstamps:
    
    HARP2_L2_MAPOL_FILENAME = "PACE_OCI.{}.L2.AER_UAA.V3_1.NRT.nc".format(tstamp)
    #https://oceandata.sci.gsfc.nasa.gov/getfile/PACE_OCI.20250910T133257.L2.AER_UAA.V3_1.NRT.nc
    try:
        print(f"{OB_DAAC_PROVISIONAL}/{HARP2_L2_MAPOL_FILENAME}")
        fs.get(f"{OB_DAAC_PROVISIONAL}/{HARP2_L2_MAPOL_FILENAME}", "/home/jovyan/oci_data/")
        print(tstamp)
    except:
        print("Download failed for tstamp", tstamp)
        pass
# %%

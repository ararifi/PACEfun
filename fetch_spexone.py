# %%
import earthaccess
fs = earthaccess.get_fsspec_https_session()
 
# %%

# BALKAN
region = (12.572136,35.345655,30.413933,49.060595) # bbox balkan region
tspan = ("2024-01-01 00:00", "2026-01-01 00:00")
tspan = ("2024-04-01 00:00", "2024-04-03 00:00")

# NOA CHLA
region=(-88.242188,4.214943,-2.988281,44.465151) # bbox NAO
tspan = ("2024-09-01 00:00", "2024-11-01 00:00")

#results = earthaccess.search_data(
#    short_name="PACE_SPEXONE_L2_AER_RTAPLAND",
#    bounding_box=region,
#    temporal=tspan,
#    cloud=False
#)

results = earthaccess.search_data(
    short_name="PACE_SPEXONE_L2_AER_RTAPOCEAN",
    temporal=tspan,
    bounding_box=region,
    granule_name="PACE_SPEXONE.*.V3_0.nc"
)
# %%
paths = earthaccess.download(results, local_path="/work/chla_world/spexone")
# %%

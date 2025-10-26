# %%
import earthaccess
fs = earthaccess.get_fsspec_https_session()
# %%PACE_OCI_L2_BGC
#region = (12.572136,35.345655,30.413933,49.060595) # bbox balkan region
region=(-88.242188,4.214943,-2.988281,44.465151) # bbox NAO
tspan = ("2024-09-01 00:00", "2024-11-01 00:00")


'''
results = earthaccess.search_data(
    short_name="PACE_OCI_L2_AER_UAA",
    bounding_box=region,
    temporal=tspan,
    granule_name="PACE_OCI.*.L2.AER_UAA.V3_1.nc"
)
'''
results = earthaccess.search_data(
    short_name="PACE_OCI_L2_BGC",
    bounding_box=region,
    temporal=tspan,
    granule_name="PACE_OCI.*.V3_1.nc"
)
# %%
paths = earthaccess.download(results, local_path="/work/chla_world/oci")



# %%

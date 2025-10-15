# %%
import earthaccess
fs = earthaccess.get_fsspec_https_session()
# %%

region = (12.572136,35.345655,30.413933,49.060595) # bbox balkan region
tspan = ("2024-01-01 00:00", "2026-01-01 00:00")


'''
results = earthaccess.search_data(
    short_name="PACE_OCI_L2_AER_UAA",
    bounding_box=region,
    temporal=tspan,
    granule_name="PACE_OCI.*.L2.AER_UAA.V3_1.nc"
)
'''
results = earthaccess.search_data(
    short_name="PACE_OCI_L3M_AER_UAA",
    bounding_box=region,
    temporal=tspan,
    granule_name="PACE_OCI.*.L3m.DAY.AER_UAA.V3_1.0p1deg.nc"
)
# %%
   
paths = earthaccess.download(results, local_path="/work/balkan/oci")



# %%

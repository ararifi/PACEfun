# %%
import earthaccess
fs = earthaccess.get_fsspec_https_session()
# %%
#earthaccess.login('login.netrc')
earthaccess.login(persist=True)
dir(earthaccess)

# %%
import ssl, certifi, fsspec

url = "https://obdaac-tea.earthdatacloud.nasa.gov/"
print("certifi bundle:", certifi.where())

# Try disabling SSL verification ONCE to confirm diagnosis (do not keep this).
fs_insecure = fsspec.filesystem("https", client_kwargs={"ssl": False})
print("Can reach endpoint (insecure):", fs_insecure.exists(url))
# %%

import os, certifi
os.environ["SSL_CERT_FILE"] = certifi.where()  # make default SSL context trust certifi CAs
# %%
certifi.where()  
# %%

 
region = (12.572136,35.345655,30.413933,49.060595) # bbox balkan region
tspan = ("2024-01-01 00:00", "2026-01-01 00:00")
tspan = ("2024-04-01 00:00", "2024-04-03 00:00")

results = earthaccess.search_data(
    short_name="PACE_SPEXONE_L2_AER_RTAPLAND",
    bounding_box=region,
    temporal=tspan,
    cloud=False
)
# %%

paths_spx = earthaccess.open(results, provider="OB.DAAC")
# %%
for g in results:
    print(g.data_links("https")) 
# %%
tstamps = [] #lists of timestamps
for i in range(len(results)):
    fname = results[i]['meta']['native-id']
    rec = fname.strip().split('.')
    tstamps.append(rec[1])
    
# %%
OB_DAAC_PROVISIONAL = "https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/"
for tstamp in tstamps:
    FILENAME = "PACE_SPEXONE.{}.L2.RTAP_LD.V3_0.nc".format(tstamp)
    #https://oceandata.sci.gsfc.nasa.gov/getfile/PACE_OCI.20250910T133257.L2.AER_UAA.V3_1.NRT.nc
    try:
        print(f"{OB_DAAC_PROVISIONAL}/{FILENAME}")
        fs.get(f"{OB_DAAC_PROVISIONAL}/{FILENAME}", "/projects/0/prjs1474/aarifi/PACEfun/balkan2")
        print(tstamp)
    except:
        print("Download failed for tstamp", tstamp)
        pass
# %%

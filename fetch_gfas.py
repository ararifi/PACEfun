#!/gpfs/work1/0/prjs1166/aarifi1/envs/pyeval/bin/python
# _v0006_emissions_inventories_GFAS_CAMS_T63_daily

# fetch.py

import cdsapi
import argparse
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor

os.environ['CDSAPI_URL'] = 'https://ads.atmosphere.copernicus.eu/api'

def download_data(spec, filename, year):
    """Function to handle data download."""
    c = cdsapi.Client()
    
    save_path = f"download/emiss_GFAS_{filename}_wildfire_{year}"
    
    date_range='2023-01-01/2024-01-01'
    if year == 2023:
        date_range = '2022-12-31/2023-12-31'
    elif year == 2024:
        date_range = '2024-01-01/2024-12-31'
    elif year == 2025:
        date_range = '2025-01-01/2025-07-31'
    else:
        print("Year not supported.")
        return
    
    
    try:
        c.retrieve(
            'cams-global-fire-emissions-gfas',
            {
                'date': date_range,
                'format': 'grib',
                'variable': spec,
            },
            save_path
        )
        print(f"Download successful for {save_path}")
    except Exception as e:
        print(f"Error downloading {save_path}: {e}")

if __name__ == "__main__":
    year = 2025  # or 2023, etc.
    species = ['wildfire_flux_of_black_carbon', 'wildfire_flux_of_dimethyl_sulfide', 
               'wildfire_flux_of_organic_carbon', 'wildfire_flux_of_sulphur_dioxide']
    filenames = ['bc', 'c2h6s', 'oc', 'so2']

    # Initialize argument parser (though not used here for input handling)
    parser = argparse.ArgumentParser(description='Download ERA5 data')

    # Using ProcessPoolExecutor to manage parallel downloads
    with ProcessPoolExecutor() as executor:
        executor.map(download_data, species, filenames, [year]*len(species))

    print("All downloads initiated or completed.")
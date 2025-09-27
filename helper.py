# HELPER
import numpy as np
import xarray as xr

def get_wv_idx(path, wv):
    dt = xr.open_datatree(path)
    return dt["sensor_band_parameters"].wavelength3d.get_index("wavelength3d").get_loc(wv)

def time_from_attr(ds):
    
    """Set the start time attribute as a dataset variable.

    Parameters
    ----------
    ds
        a dataset corresponding to a Level-2 granule
    """
    datetime = ds.attrs["time_coverage_start"].replace("Z", "")
    ds["time"] = ((), np.datetime64(datetime, "ns"))
    ds = ds.set_coords("time")
    return ds

def grid_match(path, dst_crs, dst_shape, dst_transform, var, wv_idx=None):
    geoloc_group_name = 'geolocation_data'
    
    """Reproject a Level-2 granule to match a Level-3M-ish granule."""
    dt = xr.open_datatree(path)
   
    
    if "geophysical_data" in dt:
        da = dt["geophysical_data"]
    elif "navigation_data" in dt:
        da = dt["navigation_data"]
    else:
        da = None  # or raise an error / handle differently
        return
        
        
    da = da[var]
    if wv_idx is not None:
        if "wavelength3d" in da.dims:
            da = da.sel(wavelength3d=wv_idx)
        elif "Wavelength_Used_all" in da.dims:
            da = da.sel(Wavelength_Used_all=wv_idx)
        else:
            raise KeyError("Neither 'wavelength3d' nor 'Wavelength_Used_all' found in dataset.")
    da = da.rio.set_spatial_dims("pixels_per_line", "number_of_lines")
    da = da.rio.set_crs("epsg:4326")
    da = da.rio.reproject(
        dst_crs,
        shape=dst_shape,
        transform=dst_transform,
        src_geoloc_array=(
            dt[geoloc_group_name]["longitude"],
            dt[geoloc_group_name]["latitude"],
        ),
    )
    da = da.rename({"x":"longitude", "y":"latitude"})
    return da

def crs_template(path, var, wv=None):
    datatree = xr.open_datatree(path)

    dataset = xr.merge(datatree.to_dict().values())
    dataset = dataset.set_coords(("latitude", "longitude"))
    da = dataset[var]
    if wv is not None:
        if "wavelength3d" in da.dims:
            da = da.sel(wavelength3d=wv)
        elif "Wavelength_Used_all" in da.dims:
            da = da.sel(Wavelength_Used_all=wv)
        else:
            raise KeyError("Neither 'wavelength3d' nor 'Wavelength_Used_all' found in dataset.")

    da = da.rio.set_spatial_dims("pixels_per_line", "number_of_lines")
    da = da.rio.write_crs("epsg:4326")
    da_L3M = da.rio.reproject(
        dst_crs="epsg:4326",
        src_geoloc_array=(
            da.coords["longitude"],
            da.coords["latitude"],
        ),
    )
    da_L3M = da_L3M.rename({"x":"longitude", "y":"latitude"})
    return da_L3M.rio.crs, da_L3M.rio.shape, da_L3M.rio.transform()

def grid_aligned_subset(bbox, transform, shape):
    from affine import Affine
    """
    mid: (lon, lat)
    ext: half-size in degrees (so box is lon±ext, lat±ext)
    transform: source Affine from crs_template
    shape: (height, width) from crs_template
    returns: (new_shape, new_transform, window_indices)
    """
    height, width = shape
    lon_min, lat_min, lon_max, lat_max = bbox

    # Map bounds to fractional pixel indices on the source grid
    inv = ~transform
    # upper-left of box -> (col,row)
    c0, r0 = inv * (lon_min, lat_max)
    # lower-right of box -> (col,row)
    c1, r1 = inv * (lon_max, lat_min)

    # Snap to integer pixel edges (expand to cover the whole region)
    col_off = int(np.floor(c0))
    row_off = int(np.floor(r0))
    col_max = int(np.ceil(c1))
    row_max = int(np.ceil(r1))

    # Clamp the *max* values, not the offsets

    cols = max(0, col_max - col_off)
    rows = max(0, row_max - row_off)

    # New transform is the old one shifted by the window offset
    new_transform = transform * Affine.translation(col_off, row_off)
    new_shape = (rows, cols)

    # Window indices you can use to slice arrays
    window = dict(row_off=row_off, col_off=col_off, height=rows, width=cols)
    return new_shape, new_transform, window

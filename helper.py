# HELPER
import numpy as np
import xarray as xr

# Optional plotting imports (only used by plotting helpers)
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.img_tiles as cimgt
except Exception:  # Keep helper importable even without plotting deps
    plt = None
    animation = None
    ccrs = None
    cfeature = None
    cimgt = None

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
def time_from_attr_2(ds):
    """
    Extract time from the dataset's 'time_coverage_start' attribute 
    and return it as a DataArray.

    Parameters
    ----------
    ds : xarray.Dataset
        A dataset corresponding to a Level-2 granule.
    """
    datetime = ds.attrs["time_coverage_start"].replace("Z", "")
    time_da = xr.DataArray(
        np.datetime64(datetime, "ns"),
        dims=(),
        name="time"
    )
    return time_da
def grid_match(path, dst_crs, dst_shape, dst_transform, var, wv_idx=None, quality=None):
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
    
    if quality is not None:
        quality_flag = dt["diagnostic_data"]["quality_flag"]
        da = da.where(quality_flag <= quality)
    
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


# -----------------------------
# Plotting helper functions
# -----------------------------

def _require_plotting_backends():
    if plt is None or ccrs is None:
        raise ImportError(
            "Plotting requires matplotlib and cartopy to be installed."
        )


def _add_basemap(ax, region, background="satellite", tiles_zoom=12,
                 coastlines=True, borders=True, land=True, gridlines=True):
    if background and background.lower() in {"sat", "satellite"} and cimgt is not None:
        try:
            tiler = cimgt.GoogleTiles(style="satellite")
            ax.add_image(tiler, tiles_zoom)
        except Exception:
            # Fallback silently if tiles cannot load (e.g., no network)
            pass

    ax.set_extent([region[0], region[2], region[1], region[3]], crs=ccrs.PlateCarree())

    if coastlines:
        ax.coastlines(resolution="110m", color="black", linewidth=0.6, zorder=2)
    if borders and cfeature is not None:
        ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5, zorder=5)
    if land and cfeature is not None:
        ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.2, zorder=1)
    if gridlines:
        try:
            gl = ax.gridlines(draw_labels=True, linestyle="--")
            gl.top_labels = False
            gl.right_labels = False
        except Exception:
            pass


def plot_mean_panels(
    da_list,
    region,
    titles=None,
    crosshair=None,
    cmap="viridis",
    vmin=None,
    vmax=None,
    robust=True,
    background="satellite",
    tiles_zoom=12,
    figsize=(16, 6),
    show=True,
    ncols=None,
):
    """Plot spatial mean maps for one or more DataArrays.

    Parameters
    ----------
    da_list : list[xr.DataArray] | tuple
        One or more processed DataArrays aligned to the same lon/lat grid.
    region : tuple(lon_min, lat_min, lon_max, lat_max)
        Geographic extent to display.
    titles : list[str] | None
        Panel titles; if None, indexes are used.
    crosshair : tuple(lon, lat) | None
        If provided, draw a "+" marker at this lon/lat.
    cmap, vmin, vmax, robust :
        Colormap and scaling options; if vmin/vmax are None, let xarray decide.
    background : str | None
        If "satellite", attempt to draw GoogleTiles satellite background.
    tiles_zoom : int
        Zoom level for background tiles.
    figsize : tuple
        Matplotlib figure size.
    show : bool
        Whether to call plt.show().

    Returns
    -------
    fig, axes
    """
    _require_plotting_backends()

    if not isinstance(da_list, (list, tuple)):
        da_list = [da_list]

    n = len(da_list)
    if titles is None:
        titles = [f"Panel {i+1}" for i in range(n)]

    # Normalize per-panel params
    def _as_list(val, default, N):
        if isinstance(val, (list, tuple, np.ndarray)):
            return list(val)
        return [default if val is None else val] * N

    cmap_list = _as_list(cmap, "viridis", n)
    vmin_list = _as_list(vmin, None, n)
    vmax_list = _as_list(vmax, None, n)

    # Grid layout
    if ncols is None:
        ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, subplot_kw={"projection": ccrs.PlateCarree()})
    axes = np.atleast_1d(axes).ravel().tolist()

    for i, (ax, da, title) in enumerate(zip(axes, da_list, titles)):
        _add_basemap(ax, region, background=background, tiles_zoom=tiles_zoom)
        plot_da = da
        if "time" in da.dims:
            plot_da = da.mean("time")

        plot_da.plot.imshow(
            ax=ax,
            transform=ccrs.PlateCarree(),
            robust=robust,
            cmap=cmap_list[i],
            vmin=vmin_list[i],
            vmax=vmax_list[i],
            zorder=3,
        )

        if crosshair is not None:
            cln, clt = crosshair
            ax.plot(
                cln,
                clt,
                transform=ccrs.PlateCarree(),
                marker="+",
                markersize=14,
                mew=2,
                mec="red",
                mfc="none",
                linestyle="none",
                zorder=4,
            )
        ax.set_title(title)

    # Hide any unused axes if grid has extra slots
    for j in range(len(da_list), len(axes)):
        axes[j].set_visible(False)

    if show:
        plt.show()
    return fig, axes


def animate_panels(
    da_list,
    region,
    titles=None,
    crosshair=None,
    cmap="viridis",
    vmin=None,
    vmax=None,
    interval=600,
    background=None,
    tiles_zoom=12,
    save_path=None,
    writer="pillow",
    dpi=120,
    ncols=None,
):
    """Animate time-evolving maps for one or more DataArrays side-by-side.

    Expects each DataArray to have a "time" dimension and be on the same grid.

    Parameters are similar to plot_mean_panels, with additional animation options.
    If save_path is provided, the animation is saved using the requested writer.
    Returns (fig, axes, ani).
    """
    _require_plotting_backends()
    if animation is None:
        raise ImportError("matplotlib.animation is required for animate_panels")

    if not isinstance(da_list, (list, tuple)):
        da_list = [da_list]
    n = len(da_list)
    if titles is None:
        titles = [f"Panel {i+1}" for i in range(n)]

    # Normalize per-panel params
    def _as_list(val, default, N):
        if isinstance(val, (list, tuple, np.ndarray)):
            return list(val)
        return [default if val is None else val] * N

    cmap_list = _as_list(cmap, "viridis", n)
    vmin_list = _as_list(vmin, None, n)
    vmax_list = _as_list(vmax, None, n)

    # Grid layout
    if ncols is None:
        ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 6), subplot_kw={"projection": ccrs.PlateCarree()})
    axes = np.atleast_1d(axes).ravel().tolist()

    images = []
    used_axes = 0
    for i, (ax, da, title) in enumerate(zip(axes, da_list, titles)):
        _add_basemap(ax, region, background=background, tiles_zoom=tiles_zoom)

        frame0 = da.isel(time=0)
        im = ax.imshow(
            frame0.values,
            origin="upper",
            transform=ccrs.PlateCarree(),
            extent=[region[0], region[2], region[1], region[3]],
            cmap=cmap_list[i],
            vmin=vmin_list[i],
            vmax=vmax_list[i],
            zorder=3,
        )
        # Add a colorbar per panel so it appears in the animation/GIF
        try:
            cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
            # Best-effort label from DataArray metadata
            label = da.name if hasattr(da, "name") and da.name is not None else ""
            units = da.attrs.get("units") if hasattr(da, "attrs") else None
            if units:
                label = f"{label} [{units}]".strip()
            if label:
                cbar.set_label(label)
        except Exception:
            pass
        images.append(im)
        used_axes += 1

        if crosshair is not None:
            cln, clt = crosshair
            ax.plot(
                cln,
                clt,
                transform=ccrs.PlateCarree(),
                marker="+",
                markersize=14,
                mew=2,
                mec="red",
                mfc="none",
                linestyle="none",
                zorder=4,
            )
        ax.set_title(title)

    # Hide unused axes
    for j in range(used_axes, len(axes)):
        axes[j].set_visible(False)

    # Improve layout so colorbars are not clipped in saved GIFs
    try:
        fig.tight_layout()
    except Exception:
        pass

    time_coord = da_list[0].time

    def update(frame):
        for im, da in zip(images, da_list):
            im.set_data(da.isel(time=frame).values)
        try:
            fig.suptitle(f"Time = {str(time_coord.values[frame])}")
        except Exception:
            pass
        return images

    ani = animation.FuncAnimation(
        fig, update, frames=len(time_coord), interval=interval, blit=False
    )

    if save_path is not None:
        ani.save(save_path, writer=writer, dpi=dpi)

    return fig, axes, ani


def interactive_panels(
    da_list,
    region,
    titles=None,
    crosshair=None,
    cmap="viridis",
    vmin=None,
    vmax=None,
    ncols=None,
):
    """Interactive slider to browse time for one or more DataArrays.

    Returns the ipywidgets controller. Works in notebooks.
    """
    _require_plotting_backends()
    try:
        import ipywidgets as widgets
        from ipywidgets import interact
        from IPython.display import display
    except Exception as e:
        raise ImportError("interactive_panels requires ipywidgets in a notebook") from e

    if not isinstance(da_list, (list, tuple)):
        da_list = [da_list]
    n = len(da_list)
    if titles is None:
        titles = [f"Panel {i+1}" for i in range(n)]

    def plot_frame(i):
        # Normalize per-panel params on creation time to allow arrays
        def _as_list(val, default, N):
            if isinstance(val, (list, tuple, np.ndarray)):
                return list(val)
            return [default if val is None else val] * N

        cmap_list = _as_list(cmap, "viridis", n)
        vmin_list = _as_list(vmin, None, n)
        vmax_list = _as_list(vmax, None, n)

        # Grid layout
        nc = ncols if ncols is not None else min(3, n)
        nr = int(np.ceil(n / nc))
        fig, axes = plt.subplots(nr, nc, figsize=(16, 6), subplot_kw={"projection": ccrs.PlateCarree()})
        axes = np.atleast_1d(axes).ravel().tolist()

        used_axes = 0
        for idx, (ax, da, title) in enumerate(zip(axes, da_list, titles)):
            _add_basemap(ax, region, background=None)
            im = ax.imshow(
                da.isel(time=i).values,
                origin="upper",
                transform=ccrs.PlateCarree(),
                extent=[region[0], region[2], region[1], region[3]],
                cmap=cmap_list[idx],
                vmin=vmin_list[idx],
                vmax=vmax_list[idx],
            )
            # Add a colorbar per panel for the interactive view
            try:
                cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
                label = da.name if hasattr(da, "name") and da.name is not None else ""
                units = da.attrs.get("units") if hasattr(da, "attrs") else None
                if units:
                    label = f"{label} [{units}]".strip()
                if label:
                    cbar.set_label(label)
            except Exception:
                pass
            if crosshair is not None:
                cln, clt = crosshair
                ax.plot(
                    cln,
                    clt,
                    transform=ccrs.PlateCarree(),
                    marker="+",
                    markersize=14,
                    mew=2,
                    mec="red",
                    mfc="none",
                    linestyle="none",
                    zorder=4,
                )
            ax.set_title(title)
            used_axes += 1

        for j in range(used_axes, len(axes)):
            axes[j].set_visible(False)
        try:
            fig.suptitle(f"Time = {str(da_list[0].time.values[i])}")
        except Exception:
            pass
        # Avoid colorbar being cut off in the widget output
        try:
            fig.tight_layout()
        except Exception:
            pass
        plt.show()

    slider = widgets.IntSlider(min=0, max=len(da_list[0].time) - 1, step=1, value=0)
    ctrl = interact(plot_frame, i=slider)
    try:
        display(ctrl)
    except Exception:
        pass
    return ctrl

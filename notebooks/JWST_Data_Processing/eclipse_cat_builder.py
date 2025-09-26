# -*- coding: utf-8 -*-
"""
Single-eclipse HDF5 builder (multi-visit ready).

This module creates a one-row xarray Dataset representing a single eclipse
observation ("visit") and saves it to an HDF5-based netCDF file. The schema
is designed so that multiple visits can be concatenated along the "visit"
dimension later without refactoring.

The builder:
* Computes science results from provided inputs (samples, fluxcal).
* Reads Stage-2 FITS header to gather R_* / S_* provenance and CRDS info.
* Moves visit-varying metadata into per-visit columns (data variables).
* Keeps only HLSP/target invariants in global attributes.

All lines are <= 80 chars and code follows flake8 formatting.
"""

from pathlib import Path
import numpy as np
import xarray as xr
from astropy.io import fits
from jwst.datamodels import JwstDataModel
import unicodedata
import re
import pandas as pd

# Setting default metadata that won't ever change
DOI = '10.17909/qsyr-ny68'
HLSPID = 'Rocky-Worlds'
HLSPNAME = 'Rocky Worlds DDT'
HLSP_PI = 'Néstor Espinoza, Hannah Diamond-Lowe'
HLSPLEAD = 'Taylor J. Bell'
LICENSE = 'CC BY 4.0'
LICENURL = 'https://creativecommons.org/licenses/by/4.0/'
RADESYS = 'ICRS'
TUNIT = 'mJy'
TIMESYS = 'TDB'

# Define constants for data product formatting
upperErrorPercentile = 84.135
lowerErrorPercentile = 15.865
medianPercentile = 50
eclipseDepthDecimals = 1
eclipseDepth_units = 'ppm'
eclipseTimeDecimals = 5
timeOffset = 2400000.5
time_units = 'BJD_TDB'
absoluteFluxDecimals = 5
lightcurveDecimals = 7
timeDecimals = 7
lightcurve_units = 'None (normalized)'


def _da1(value, visit_idx, units=None):
    """
    Create a 1-element DataArray along the ``visit`` dimension.

    Parameters
    ----------
    value : Any
        Value to place in the single-element array.
    visit_idx : int
        Visit index to use as the coordinate value.
    units : str or None, optional
        Units to attach to the DataArray (stored in ``attrs['units']``).

    Returns
    -------
    xarray.DataArray
        DataArray with shape ``(visit: 1)`` and optional units attribute.
    """
    da = xr.DataArray(
        np.array([value]),
        coords={'visit': [visit_idx]},
        dims=['visit'],
    )
    if units:
        da.attrs['units'] = units
    return da


def _derive_visit_id_string(file_ids):
    """
    Parse JWST segment names to collect one or more VISIT_ID strings.

    Parameters
    ----------
    file_ids : list of str
        Base names like
        ``jw09235001001_03101_00001-seg001_mirimage`` (no extension).

    Returns
    -------
    str
        Comma-separated, numerically sorted unique VISIT_IDs, each an
        11-digit string (e.g., ``'09235001001,09235002001'``). Returns
        an empty string if none are found.

    Notes
    -----
    The parser looks for ``^jw(\\d{11})_`` at the start of each base name.
    """
    pat = re.compile(r'^jw(\d{11})_')
    ids = []
    for fid in file_ids:
        base = Path(fid).name
        m = pat.match(base)
        if m:
            ids.append(m.group(1))
    uniq = sorted(set(ids), key=int)
    return ','.join(uniq)


def read_stage2_r_s_meta(stage2_fits_path):
    """
    Read Stage-2 FITS header and extract provenance and basic context.

    Parameters
    ----------
    stage2_fits_path : str
        Path to the Stage-2 FITS file (calints product).

    Returns
    -------
    r_keys : dict
        Mapping of ``R_*`` keywords to string values from the header.
    s_keys : dict
        Mapping of ``S_*`` keywords to string values from the header.
    crds_ver : str or None
        CRDS software version (``CRDS_VER``) if present.
    crds_ctx : str or None
        CRDS context (``CRDS_CTX`` or ``CRDS_PMAP``) if present.
    filt : str
        Filter name (``FILTER``) or empty string if not present.
    subarray : str
        Subarray name (``SUBARRAY``) or empty string if not present.
    date_obs : numpy.datetime64
        Combined ``DATE-OBS`` and ``TIME-OBS`` as ``datetime64[ns]``.

    Notes
    -----
    Only the primary header (HDU 0) is inspected. Values are returned as
    strings when available; missing cards yield empty strings or ``None``.
    """
    with fits.open(stage2_fits_path, memmap=False) as hdul:
        hdr = hdul[0].header
        keys = list(hdr.keys())
        r_keys = {k: hdr[k] for k in keys if isinstance(k, str) and k[:2] == 'R_'}
        s_keys = {k: hdr[k] for k in keys if isinstance(k, str) and k[:2] == 'S_'}
        crds_ver = hdr.get('CRDS_VER')
        crds_ctx = hdr.get('CRDS_CTX', hdr.get('CRDS_PMAP'))
        filt = hdr.get('FILTER', '') or ''
        subarray = hdr.get('SUBARRAY', '') or ''
        date = str(hdr.get('DATE-OBS', '')).strip()
        time = str(hdr.get('TIME-OBS', '')).strip() or '00:00:00'
    if date:
        try:
            date_obs = np.datetime64(f'{date}T{time}')
        except Exception:
            date_obs = np.datetime64('NaT')
    else:
        date_obs = np.datetime64('NaT')
    return r_keys, s_keys, crds_ver, crds_ctx, filt, subarray, date_obs


def build_single_eclipse(
    stage2_fits,
    stage3_specdata,
    stage4cal,
    stage5_samples,
    visit,
    STAR,
    PLANET,
    SRC_DOI,
    HLSPVER,
):
    """
    Build a one-row (visit) Dataset with visit-varying fields as columns.

    This function loads all required inputs from file paths. It computes the
    eclipse depth, timing, and absolute flux with uncertainties, collects
    per-visit provenance from the Stage-2 FITS header (``R_*``, ``S_*``, and
    ``CRDS_*``), and stores HLSP invariants in global attributes.

    Parameters
    ----------
    stage2_fits : str
        Path to the Stage 2 FITS file.
    stage3_specdata : str
        Path to Stage 3 product providing per-segment file list and versions.
    stage4cal : str
        Path to Stage 4 flux calibration product holding stellar flux and
        error.
    stage5_samples : str
        Path to Stage 5 posterior samples with ``fp`` and ``t_secondary``
        variables.
    visit : int
        Eclipse index you assign (1, 2, ...) used for the ``visit`` coord
        and in output filenames. This is distinct from JWST VISIT_ID(s).
    STAR : str
        Host star identifier (per visit).
    PLANET : str
        Planet identifier (per visit).
    SRC_DOI : str
        DOI for the source data specific to this visit.
    HLSPVER : str
        HLSP version string to store in the dataset attributes.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with a single row along ``visit`` that contains results,
        visit-varying metadata as columns, and global HLSP invariants.

    Notes
    -----
    The output is designed for later concatenation along the ``visit``
    dimension using :func:`xarray.concat`. The ``VISIT_ID`` column is
    derived from Stage-3 segment names and may contain multiple IDs
    (comma-separated) if the eclipse spans several JWST observations.
    """
    # Load inputs (the LC file path is accepted but not used nor stored)
    spec_ds = xr.load_dataset(stage3_specdata)
    fluxcal = xr.load_dataset(stage4cal)
    samples = xr.load_dataset(stage5_samples)
    dm = JwstDataModel(stage2_fits)

    # FITS header provenance and context
    (
        r_from_fits,
        s_from_fits,
        crds_ver_hdr,
        crds_ctx_hdr,
        filt_hdr,
        subarray_hdr,
        date_obs,
    ) = read_stage2_r_s_meta(stage2_fits)

    # Prefer header values when available
    instrume = dm.meta.instrument.name
    filter_name = filt_hdr or dm.meta.instrument.filter
    subarray = subarray_hdr or getattr(dm.meta.subarray, 'name', '')
    telescop = dm.meta.telescope
    observat = dm.meta.telescope

    # Pointing / program context
    proposid = dm.meta.observation.program_number
    ra_targ = dm.meta.target.ra
    dec_targ = dm.meta.target.dec

    # Segment list to FILE_ID (comma-separated string)
    file_ids = []
    if hasattr(spec_ds, 'segment_list'):
        for seg in spec_ds.segment_list:
            name = Path(seg).name
            file_ids.append(name.split('_calints')[0])
    file_id_str = ','.join(file_ids)

    # Derive VISIT_IDs from segment base names
    visit_id_str = _derive_visit_id_string(file_ids)

    # Exposure timing
    mjd_beg = float(dm.meta.exposure.start_time_tdb)
    mjd_mid = float(dm.meta.exposure.mid_time_tdb)
    mjd_end = float(dm.meta.exposure.end_time_tdb)
    xposure = float(dm.meta.exposure.effective_exposure_time)

    # Software versions
    cal_ver = str(dm.meta.calibration_software_version)
    pipe_line = str(getattr(spec_ds, 'data_format', ''))
    pipe_ver = str(getattr(spec_ds, 'version', ''))
    crds_ver = str(
        crds_ver_hdr or getattr(dm.meta.ref_file.crds, 'sw_version', '')
    )
    crds_ctx = str(
        crds_ctx_hdr or getattr(dm.meta.ref_file.crds, 'context_used', '')
    )

    # Science results: eclipse depth (ppm)
    depth_vals = samples.fp.values * 1e6
    d_med = np.round(np.median(depth_vals), eclipseDepthDecimals)
    d_std = np.round(np.std(depth_vals), eclipseDepthDecimals)
    d_p = np.percentile(
        depth_vals,
        [lowerErrorPercentile, medianPercentile, upperErrorPercentile],
    )
    d_up = np.round(d_p[2] - d_p[1], eclipseDepthDecimals)
    d_lo = np.round(d_p[1] - d_p[0], eclipseDepthDecimals)

    # Science results: eclipse time
    t_vals = samples.t_secondary.values + timeOffset
    t_med = np.round(np.median(t_vals), eclipseTimeDecimals)
    t_std = np.round(np.std(t_vals), eclipseTimeDecimals)
    t_p = np.percentile(
        t_vals,
        [lowerErrorPercentile, medianPercentile, upperErrorPercentile],
    )
    t_up = np.round(t_p[2] - t_p[1], eclipseTimeDecimals)
    t_lo = np.round(t_p[1] - t_p[0], eclipseTimeDecimals)

    # Science results: absolute flux (with systematic inflation)
    stellar_flux = float(fluxcal.ecl_flux.data[0][0])
    ferr = float(fluxcal.ecl_ferr.data[0][0])
    sys_err = (0.0048 * stellar_flux) ** 2 + (0.0045 * stellar_flux) ** 2
    stellar_flux_err = np.sqrt(ferr**2 + sys_err)
    f_med = np.round(stellar_flux, absoluteFluxDecimals)
    f_err = np.round(stellar_flux_err, absoluteFluxDecimals)

    # Build Dataset (one row)
    v = int(visit)
    ds = xr.Dataset()

    # Results with units (via helper to avoid duplication)
    ds['eclipseDepth'] = _da1(d_med, v, units=eclipseDepth_units)
    ds['eclipseDepthError'] = _da1(d_std, v, units=eclipseDepth_units)
    ds['eclipseDepthUpperError'] = _da1(d_up, v, units=eclipseDepth_units)
    ds['eclipseDepthLowerError'] = _da1(d_lo, v, units=eclipseDepth_units)

    ds['eclipseTime'] = _da1(t_med, v, units=time_units)
    ds['eclipseTimeError'] = _da1(t_std, v, units=time_units)
    ds['eclipseTimeUpperError'] = _da1(t_up, v, units=time_units)
    ds['eclipseTimeLowerError'] = _da1(t_lo, v, units=time_units)

    ds['absFlux'] = _da1(f_med, v, units=TUNIT)
    ds['absFluxError'] = _da1(f_err, v, units=TUNIT)
    ds['absFluxUpperError'] = _da1(f_err, v, units=TUNIT)
    ds['absFluxLowerError'] = _da1(f_err, v, units=TUNIT)

    # Only "visit" is a coordinate
    ds = ds.assign_coords(visit=[v])

    # Visit context as DATA VARIABLES (not coordinates)
    ds['date_obs'] = _da1(date_obs, v)
    ds['filter'] = _da1(str(filter_name), v)
    ds['subarray'] = _da1(str(subarray), v)

    # Visit-varying metadata as columns
    ds['SRC_DOI'] = _da1(str(SRC_DOI), v)
    ds['VISIT_ID'] = _da1(visit_id_str, v)       # was 'VISIT'
    ds['FILE_ID'] = _da1(file_id_str, v)
    ds['MJD_BEG'] = _da1(mjd_beg, v)
    ds['MJD_MID'] = _da1(mjd_mid, v)
    ds['MJD_END'] = _da1(mjd_end, v)
    ds['XPOSURE'] = _da1(xposure, v)
    ds['CAL_VER'] = _da1(cal_ver, v)
    ds['CRDS_VER'] = _da1(crds_ver, v)
    ds['CRDS_CTX'] = _da1(crds_ctx, v)
    ds['PIPELINE'] = _da1(pipe_line, v)
    ds['PIPE_VER'] = _da1(pipe_ver, v)

    # Per-visit provenance from FITS header (R_* and S_* cards)
    for key, val in r_from_fits.items():
        ds[key] = _da1(str(val), v)
    for key, val in s_from_fits.items():
        ds[key] = _da1(str(val), v)

    # Global, visit-invariant attributes (constants + visit context)
    ds.attrs.update(
        {
            'HLSPVER': HLSPVER,
            'HLSPID': HLSPID,
            'HLSPNAME': HLSPNAME,
            'HLSP_PI': HLSP_PI,
            'HLSPLEAD': HLSPLEAD,
            'DOI': DOI,
            'STAR': STAR,
            'PLANET': PLANET,
            'HLSPTARG': PLANET,
            'OBSERVAT': observat,
            'TELESCOP': telescop,
            'INSTRUME': instrume,
            'RADESYS': RADESYS,
            'TIMESYS': TIMESYS,
            'TUNIT': TUNIT,
            'LICENSE': LICENSE,
            'LICENURL': LICENURL,
            'RA_TARG': ra_targ,
            'DEC_TARG': dec_targ,
            'PROPOSID': proposid,
        }
    )

    return ds


def save_single_eclipse_hdf5(ds, out_dir='.'):
    """
    Save the dataset to an HDF5-based netCDF file with a standard filename.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset produced by the builder. Must contain a single visit row.
    out_dir : str, optional
        Directory where the file will be written. Defaults to current dir.

    Returns
    -------
    out_path : str
        Path to the written file on disk.
    """
    instrume = ds.attrs['INSTRUME']
    planet = ds.attrs['PLANET']
    visit = int(np.asarray(ds['visit'].values).item())
    filt = str(ds['filter'].values[0])
    hlspver = ds.attrs['HLSPVER']

    planet_fn = ''.join(planet.lower().split())
    out_name = (
        f'hlsp_rocky-worlds_jwst_{instrume.lower()}_{planet_fn}-'
        f'obs{visit:03d}_{filt.lower()}_v{hlspver.lower()}_'
        f'eclipse-cat.h5'
    )
    out_path = str(Path(out_dir) / out_name)

    enc = {
        name: {'zlib': True, 'complevel': 4}
        for name in ds.data_vars
        if np.issubdtype(ds[name].dtype, np.floating)
    }
    ds.to_netcdf(
        out_path,
        engine='h5netcdf',
        invalid_netcdf=True,
        encoding=enc,
        mode='w',
    )
    return out_path

###############################################################################
# Code to combine multiple single-eclipse datasets into one dataset, for use
# at RWDDT checkpoints.

def _ensure_var(ds, name, vtype, visit_idx):
    """
    Ensure a variable exists with the requested type, creating if needed.

    Parameters
    ----------
    ds : xarray.Dataset
        Single-visit dataset to modify in place.
    name : str
        Variable name to ensure.
    vtype : {'float', 'str', 'datetime'}
        Desired variable type.
    visit_idx : int
        Visit index for the single row.

    Returns
    -------
    xarray.Dataset
        The input dataset (modified) is also returned for chaining.
    """
    if name in ds:
        return ds
    if vtype == 'float':
        ds[name] = _da1(np.nan, visit_idx)
    elif vtype == 'datetime':
        ds[name] = _da1(np.datetime64('NaT'), visit_idx)
    else:
        ds[name] = _da1('', visit_idx)
    return ds


def concat_eclipse_visits(datasets):
    """
    Concatenate multiple single-visit eclipse datasets along ``visit``.

    Parameters
    ----------
    datasets : list of xarray.Dataset
        Each item must be a dataset produced by
        :func:`build_single_eclipse_dataset_multi_ready` and contain exactly
        one row along the ``visit`` dimension.

    Returns
    -------
    combined : xarray.Dataset
        Dataset with one row per input visit. Variables are the union of all
        inputs; missing values are filled with suitable blanks.

    Notes
    -----
    - Numeric visit fields are kept as float64 and filled with NaN.
    - Datetime fields use ``datetime64[ns]`` and are filled with NaT.
    - All provenance text (``R_*``, ``S_*``, and ``CRDS_*``) are strings.
    """
    if not datasets:
        raise ValueError("No datasets provided for concatenation.")

    for ds in datasets:
        if 'visit' not in ds.coords:
            raise ValueError('Each dataset must have a "visit" coordinate.')
        if ds.dims.get('visit', 0) != 1:
            raise ValueError('Each dataset must be a single-visit dataset.')

    # Union of variable names across all visits
    all_vars = set()
    for ds in datasets:
        all_vars.update(ds.data_vars)

    # Known numeric and datetime visit fields
    numeric_vars = {
        'eclipseDepth', 'eclipseDepthError', 'eclipseDepthUpperError',
        'eclipseDepthLowerError', 'eclipseTime', 'eclipseTimeError',
        'eclipseTimeUpperError', 'eclipseTimeLowerError', 'absFlux',
        'absFluxError', 'absFluxUpperError', 'absFluxLowerError', 'MJD_BEG',
        'MJD_MID', 'MJD_END', 'XPOSURE',
    }
    datetime_vars = {'date_obs'}

    def _kind(name):
        if name in numeric_vars:
            return 'float'
        if name in datetime_vars:
            return 'datetime'
        # strings: VISIT_ID, FILTER/SUBARRAY, R_*/S_*, CRDS_*, FILE_ID, etc.
        if name in {'VISIT_ID', 'filter', 'subarray', 'FILE_ID',
                    'CRDS_VER', 'CRDS_CTX', 'PIPELINE', 'PIPE_VER',
                    'SRC_DOI'}:
            return 'str'
        if name.startswith('R_') or name.startswith('S_'):
            return 'str'
        return 'str'

    # Normalize variable sets across inputs
    norm = []
    for ds in datasets:
        # get Python int visit index from coordinate
        v = int(np.asarray(ds['visit'].values).item())
        for name in sorted(all_vars):
            ds = _ensure_var(ds, name, _kind(name), v)
        norm.append(ds)

    combined = xr.concat(norm, dim='visit', join='outer')

    # Sort rows by visit index
    order = np.argsort(combined['visit'].values)
    combined = combined.isel(visit=order)

    return combined


def save_multi_eclipse_hdf5(datasets, checkpoint, out_dir='.', hlspver=None):
    """
    Save multiple visits as a single HDF5 file named with ``checkpoint##``.

    Parameters
    ----------
    datasets : list of xarray.Dataset
        List of single-visit datasets created by the builder. Each must have
        exactly one ``visit`` row.
    checkpoint : int
        Integer checkpoint number used in the filename. It is rendered as
        a two-digit, zero-padded integer (e.g., 1 -> ``checkpoint01``).
    out_dir : str, optional
        Directory where the file is written. Defaults to current directory.
    hlspver : str or None, optional
        HLSP version to embed in the filename. If ``None``, the function
        uses the ``HLSPVER`` attribute from the first dataset.

    Returns
    -------
    combined : xarray.Dataset
        The concatenated multi-visit dataset that was written to disk.
    out_path : str
        Path to the written file.

    Raises
    ------
    ValueError
        If input datasets are empty, inconsistent in key invariants, or
        if HLSP version cannot be determined.

    Notes
    -----
    The output filename pattern is:

    ``hlsp_rocky-worlds_jwst_{instrume}_{planet}-checkpoint##_v{ver}_``
    ``eclipse-cat.h5``

    The filter and per-visit light-curve paths are not part of the name.
    """
    if not datasets:
        raise ValueError("No datasets provided.")

    # Verify visit-invariant attrs are consistent across inputs
    keys = ['INSTRUME', 'PLANET']
    vals = {k: set() for k in keys}
    for ds in datasets:
        for k in keys:
            vals[k].add(str(ds.attrs.get(k, '')))
    for k in keys:
        if len(vals[k]) != 1:
            raise ValueError(f"Inconsistent attribute across inputs: {k}")

    instrume = list(vals['INSTRUME'])[0]
    planet = list(vals['PLANET'])[0]

    # Determine HLSP version for filename
    ver = hlspver or str(datasets[0].attrs.get('HLSPVER', '')).strip()
    if not ver:
        raise ValueError("HLSP version not provided and not found in attrs.")

    # Concat visits
    combined = concat_eclipse_visits(datasets)

    # Build filename with checkpoint## (two digits)
    planet_fn = ''.join(planet.lower().split())
    ckpt = f"checkpoint{int(checkpoint):02d}"
    out_name = (
        f"hlsp_rocky-worlds_jwst_{instrume.lower()}_{planet_fn}-"
        f"{ckpt}_v{ver.lower()}_eclipse-cat.h5"
    )
    out_path = str(Path(out_dir) / out_name)

    # Compression for floats
    enc = {
        name: {'zlib': True, 'complevel': 4}
        for name in combined.data_vars
        if np.issubdtype(combined[name].dtype, np.floating)
    }

    combined.to_netcdf(
        out_path,
        engine='h5netcdf',
        invalid_netcdf=True,
        encoding=enc,
        mode='w',
    )
    return combined, out_path

###############################################################################
# Code to convert Xarray into FITS for MAST automatic metadata ingest


def _to_ascii(text):
    """
    Normalize a string to plain ASCII for FITS header compatibility.

    Parameters
    ----------
    text : Any
        Value to convert to ASCII. Non-strings are returned unchanged.

    Returns
    -------
    str or Any
        ASCII-only string if input was a string; otherwise the original
        value.
    """
    if not isinstance(text, str):
        return text
    norm = unicodedata.normalize('NFKD', text)
    return norm.encode('ascii', 'ignore').decode('ascii')


def _set_card(hdr, key, value, comment=None):
    """
    Set a FITS header card with ASCII-safe text conversion.

    Parameters
    ----------
    hdr : astropy.io.fits.Header
        Header to be modified.
    key : str
        FITS keyword to write.
    value : Any
        Value to write. Strings are ASCII-normalized.
    comment : str or None, optional
        Optional comment for the card.
    """
    if value is None:
        return
    if isinstance(value, str):
        value = _to_ascii(value)
    hdr[key] = (value, None if comment is None else _to_ascii(comment))


def _collect_visit_meta(ds, i):
    """
    Collect per-visit metadata from dataset for visit index ``i``.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset produced by the builder.
    i : int
        Row index along ``visit``.

    Returns
    -------
    meta : dict
        Mapping of per-visit keys to scalars (strings/floats).
    """
    def s(name):
        return None if name not in ds else ds[name].values[i].item()
    out = {
        'SRC_DOI': s('SRC_DOI'),
        'VISIT_ID': s('VISIT_ID'),
        'FILE_ID': s('FILE_ID'),
        'MJD_BEG': s('MJD_BEG'),
        'MJD_MID': s('MJD_MID'),
        'MJD_END': s('MJD_END'),
        'XPOSURE': s('XPOSURE'),
        'CAL_VER': s('CAL_VER'),
        'CRDS_VER': s('CRDS_VER'),
        'CRDS_CTX': s('CRDS_CTX'),
        'PIPELINE': s('PIPELINE'),
        'PIPE_VER': s('PIPE_VER'),
        'FILTER': s('filter'),
        'SUBARRAY': s('subarray'),
    }
    # date_obs may be datetime64 -> render as ISO string
    if 'date_obs' in ds:
        val = ds['date_obs'].values[i]
        out['DATE-OBS'] = (np.datetime_as_string(val)
                          if not np.isnat(val) else '')
    return out


def _r_s_pairs_for_visit(ds, i):
    """
    Return (R_*, S_*) pairs for visit ``i`` from dataset variables.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset produced by the builder.
    i : int
        Row index along ``visit``.

    Returns
    -------
    r_pairs : list of tuple
        List of (key, str_value) for R_* cards.
    s_pairs : list of tuple
        List of (key, str_value) for S_* cards.
    """
    r_pairs, s_pairs = [], []
    for name in ds.data_vars:
        if name.startswith('R_') or name.startswith('S_'):
            val = ds[name].values[i].item()
            pair = (name, '' if val is None else str(val))
            if name.startswith('R_'):
                r_pairs.append(pair)
            else:
                s_pairs.append(pair)
    r_pairs.sort()
    s_pairs.sort()
    return r_pairs, s_pairs


def _build_measurements_hdu(ds):
    """
    Build the MEASUREMENTS binary table from dataset variables.

    Parameters
    ----------
    ds : xarray.Dataset
        Single- or multi-visit dataset produced by the builder.

    Returns
    -------
    hdu : astropy.io.fits.BinTableHDU
        Table with one row per visit containing numeric results and a few
        context strings.

    Notes
    -----
    This rigid version assumes all required variables exist in ``ds``.
    """
    n = ds.sizes['visit']

    # Fixed units from variable attrs (no presence checks)
    units = {
        'ECLIPSEDEPTH': ds['eclipseDepth'].attrs.get('units', ''),
        'ECLIPSEDEPTHERROR': ds['eclipseDepthError'].attrs.get('units', ''),
        'ECLIPSEDEPTHUPPERERROR':
        ds['eclipseDepthUpperError'].attrs.get('units', ''),
        'ECLIPSEDEPTHLOWERERROR':
        ds['eclipseDepthLowerError'].attrs.get('units', ''),
        'ECLIPSETIME': ds['eclipseTime'].attrs.get('units', ''),
        'ECLIPSETIMEERROR': ds['eclipseTimeError'].attrs.get('units', ''),
        'ECLIPSETIMEUPPERERROR':
        ds['eclipseTimeUpperError'].attrs.get('units', ''),
        'ECLIPSETIMELOWERERROR':
        ds['eclipseTimeLowerError'].attrs.get('units', ''),
        'ABSFLUX': ds['absFlux'].attrs.get('units', ''),
        'ABSFLUXERROR': ds['absFluxError'].attrs.get('units', ''),
        'ABSFLUXUPPERERROR': ds['absFluxUpperError'].attrs.get('units', ''),
        'ABSFLUXLOWERERROR': ds['absFluxLowerError'].attrs.get('units', ''),
        'MJD_BEG': 'BMJD_TDB',
        'MJD_MID': 'BMJD_TDB',
        'MJD_END': 'BMJD_TDB',
        'XPOSURE': 's',
    }

    # Helper for float columns (format 'D' = float64)
    def f(name):
        up = name.upper()
        arr = ds[name].values.astype(np.float64)
        unit = _to_ascii(units.get(up, '')) or None
        return fits.Column(name=up, array=arr, format='D', unit=unit)

    cols = [
        f('eclipseDepth'),
        f('eclipseDepthError'),
        f('eclipseDepthUpperError'),
        f('eclipseDepthLowerError'),
        f('eclipseTime'),
        f('eclipseTimeError'),
        f('eclipseTimeUpperError'),
        f('eclipseTimeLowerError'),
        f('absFlux'),
        f('absFluxError'),
        f('absFluxUpperError'),
        f('absFluxLowerError'),
        f('MJD_BEG'),
        f('MJD_MID'),
        f('MJD_END'),
        f('XPOSURE'),
    ]

    # Helper for string columns (variable width up to 64)
    def scol(name, outname):
        vals = []
        if name == 'DATEOBS':
            for i in range(n):
                t = ds['date_obs'].values[i]
                s = np.datetime_as_string(t) if not np.isnat(t) else ''
                vals.append(_to_ascii(s))
        else:
            for i in range(n):
                v = ds[name].values[i].item()
                vals.append(_to_ascii('' if v is None else str(v)))
        width = max(1, min(64, max(len(v) for v in vals)))
        return fits.Column(name=outname, array=np.asarray(vals),
                           format=f'A{width}')

    cols += [
        scol('VISIT_ID', 'VISIT_ID'),
        scol('filter', 'FILTER'),
        scol('subarray', 'SUBARRAY'),
        scol('DATEOBS', 'DATEOBS'),
    ]

    return fits.BinTableHDU.from_columns(cols, name='MEASUREMENTS')


def _primary_hdu_from_attrs(ds, visit_index=None):
    """
    Construct a PRIMARY HDU from dataset attributes (+ optional visit meta).

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset produced by the builder.
    visit_index : int or None, optional
        If given, include per-visit metadata for that row in PRIMARY.
        If ``None``, include only visit-invariant HLSP attributes.

    Returns
    -------
    hdu : astropy.io.fits.PrimaryHDU
        Primary HDU with populated header and no data.
    """
    hdr = fits.Header()
    # HLSP/target invariants
    for key in [
        'HLSPVER', 'HLSPID', 'HLSPNAME', 'HLSP_PI', 'HLSPLEAD', 'DOI',
        'STAR', 'PLANET', 'HLSPTARG', 'OBSERVAT', 'TELESCOP', 'INSTRUME',
        'RADESYS', 'TIMESYS', 'TUNIT', 'LICENSE', 'LICENURL', 'RA_TARG',
        'DEC_TARG', 'PROPOSID',
    ]:
        _set_card(hdr, key, ds.attrs.get(key))
    # Single-visit: include per-visit metadata in PRIMARY
    if visit_index is not None:
        meta = _collect_visit_meta(ds, visit_index)
        for k, v in meta.items():
            _set_card(hdr, k, v)
        # Also include all R_*/S_* cards
        r_pairs, s_pairs = _r_s_pairs_for_visit(ds, visit_index)
        for k, v in r_pairs + s_pairs:
            _set_card(hdr, k, v)
    return fits.PrimaryHDU(header=hdr)


def _obs_hdu_for_visit(ds, i):
    """
    Build a header-only table HDU named OBS### for visit ``i``.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset produced by the builder.
    i : int
        Visit row index.

    Returns
    -------
    hdu : astropy.io.fits.BinTableHDU
        Empty binary table with header cards carrying visit metadata and
        provenance. Name is ``OBS###`` (1-based, zero-padded).
    """
    # Empty table; header-only payload
    hdu = fits.BinTableHDU.from_columns([], nrows=0)
    hdu.header['EXTNAME'] = f'OBS{(i+1):03d}'
    meta = _collect_visit_meta(ds, i)
    for k, v in meta.items():
        _set_card(hdu.header, k, v)
    r_pairs, s_pairs = _r_s_pairs_for_visit(ds, i)
    for k, v in r_pairs + s_pairs:
        _set_card(hdu.header, k, v)
    return hdu


def hdf5_single_to_fits(h5_path):
    """
    Convert a single-visit HDF5 eclipse-cat file to a FITS file.

    Parameters
    ----------
    h5_path : str
        Path to a single-visit HDF5 produced by the builder.

    Returns
    -------
    out_path : str
        The path written.

    Raises
    ------
    ValueError
        If the input dataset is not a single-visit dataset.
    """
    ds = xr.load_dataset(h5_path)
    n = ds.dims.get('visit', 0)
    if n != 1:
        raise ValueError('Input is not a single-visit dataset.')
    phdu = _primary_hdu_from_attrs(ds, visit_index=0)
    meas = _build_measurements_hdu(ds)
    hdul = fits.HDUList([phdu, meas])
    out_path = str(Path(h5_path).with_suffix('.fits'))
    hdul.writeto(out_path, overwrite=True)
    return out_path


def hdf5_checkpoint_to_fits(h5_path):
    """
    Convert a multi-visit checkpoint HDF5 file to a FITS file.

    Parameters
    ----------
    h5_path : str
        Path to a multi-visit HDF5 produced by concatenation.

    Returns
    -------
    out_path : str
        The path written.

    Raises
    ------
    ValueError
        If the input dataset has fewer than two visits.
    """
    ds = xr.load_dataset(h5_path)
    n = ds.dims.get('visit', 0)
    if n < 2:
        raise ValueError('Input is not multi-visit (>=2 required).')
    phdu = _primary_hdu_from_attrs(ds, visit_index=None)
    hdus = [phdu]
    # One header-only OBS### per visit
    for i in range(n):
        hdus.append(_obs_hdu_for_visit(ds, i))
    # Measurements table with all visits
    hdus.append(_build_measurements_hdu(ds))
    out_path = str(Path(h5_path).with_suffix('.fits'))
    fits.HDUList(hdus).writeto(out_path, overwrite=True)
    return out_path


###############################################################################
# Code to make and save the light curve xarray/HDF5 datasets

def _series1(values, time, visit_idx, units=None, decimals=None):
    """
    Create a 1xN (visit, time) DataArray for a light-curve series.

    Parameters
    ----------
    values : array-like
        1-D time series values.
    time : array-like
        1-D time coordinate (same length as ``values``).
    visit_idx : int
        Visit index to use for the ``visit`` coordinate.
    units : str or None, optional
        Units stored in ``attrs['units']``.
    decimals : int or None, optional
        If given, round the data to this many decimals.

    Returns
    -------
    xarray.DataArray
        DataArray with dims ``('time',)`` expanded to ``('visit','time')``.
    """
    arr = np.asarray(values)
    if decimals is not None:
        arr = np.round(arr, decimals)
    da = xr.DataArray(arr, coords={'time': time}, dims=['time'])
    da = da.expand_dims(visit=[visit_idx])
    if units is not None:
        da.attrs['units'] = units
    return da


def _derive_visit_id_string(file_ids):
    r"""
    Parse JWST segment names to collect one or more VISIT_ID strings.

    Parameters
    ----------
    file_ids : list of str
        Base names like
        ``jw09235001001_03101_00001-seg001_mirimage`` (no extension).

    Returns
    -------
    str
        Comma-separated, numerically sorted unique VISIT_IDs, each an
        11-digit string (e.g., ``'09235001001,09235002001'``). Returns
        an empty string if none are found.

    Notes
    -----
    The parser looks for ``^jw(\d{11})_`` at the start of each base name.
    """
    pat = re.compile(r'^jw(\d{11})_')
    ids = []
    for fid in file_ids:
        base = Path(fid).name
        m = pat.match(base)
        if m:
            ids.append(m.group(1))
    uniq = sorted(set(ids), key=int)
    return ','.join(uniq)


def build_lightcurve_dataset(
    stage2_fits,
    stage3_specdata,
    stage4_lcdata,
    stage5_fit_table,
    visit,
    STAR,
    PLANET,
    SRC_DOI,
    HLSPVER,
):
    """
    Build a single-visit light-curve Dataset (multi-visit ready).

    The time axis is taken from the full Stage-4 LCData product. Series
    from the Stage-5 fit table are aligned onto that axis, with NaNs
    where the fit dropped integrations. Stage-4 raw flux is scaled by
    the same multiplicative divisor used in Stage-5, and its errors are
    divided by that same factor.

    Parameters
    ----------
    stage2_fits : str
        Path to the Stage 2 FITS file.
    stage3_specdata : str
        Path to Stage 3 product with ``segment_list``.
    stage4_lcdata : str
        Path to Stage 4 light-curve HDF5 (full time and centroids).
    stage5_fit_table : str
        Path to Stage 5 table (text) with columns:
        ``time``, ``lcdata``, ``lcerr``, ``astrophysical model``,
        ``model``, ``GP``, ``residuals``.
    visit : int
        Eclipse index you assign (1, 2, ...) used for the ``visit`` coord
        and in output filenames. Not the JWST VISIT_ID.
    STAR : str
        Host star identifier (per visit).
    PLANET : str
        Planet identifier (per visit).
    SRC_DOI : str
        DOI for the source data specific to this visit.
    HLSPVER : str
        HLSP version string to store in the dataset attributes.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with dims ``(visit, time)`` for series and per-visit
        metadata as 1× columns. Ready for later concatenation on
        ``visit``.
    """
    # Load inputs
    spec_ds = xr.load_dataset(stage3_specdata)
    lc = xr.load_dataset(stage4_lcdata)
    dm = JwstDataModel(stage2_fits)
    fit = pd.read_csv(stage5_fit_table, sep=r'\s+', comment='#')

    # Provenance/context from Stage-2 header
    (
        r_from_fits,
        s_from_fits,
        crds_ver_hdr,
        crds_ctx_hdr,
        filt_hdr,
        subarray_hdr,
        date_obs,
    ) = read_stage2_r_s_meta(stage2_fits)

    instrume = dm.meta.instrument.name
    filter_name = filt_hdr or dm.meta.instrument.filter
    subarray = subarray_hdr or getattr(dm.meta.subarray, 'name', '')
    telescop = dm.meta.telescope
    observat = dm.meta.telescope

    proposid = dm.meta.observation.program_number
    ra_targ = dm.meta.target.ra
    dec_targ = dm.meta.target.dec

    # FILE_ID list and VISIT_IDs derived from Stage-3 segment_list
    base_ids = []
    if hasattr(spec_ds, 'segment_list'):
        for seg in spec_ds.segment_list:
            name = Path(seg).name
            base_ids.append(name.split('_calints')[0])
    file_id_str = ','.join(base_ids)
    visit_id_str = _derive_visit_id_string(base_ids)

    # Exposure timing
    mjd_beg = float(dm.meta.exposure.start_time_tdb)
    mjd_mid = float(dm.meta.exposure.mid_time_tdb)
    mjd_end = float(dm.meta.exposure.end_time_tdb)
    xposure = float(dm.meta.exposure.effective_exposure_time)

    # Software versions
    cal_ver = str(dm.meta.calibration_software_version)
    pipe_line = str(getattr(spec_ds, 'data_format', ''))
    pipe_ver = str(getattr(spec_ds, 'version', ''))
    crds_ver = str(
        crds_ver_hdr or getattr(dm.meta.ref_file.crds, 'sw_version', '')
    )
    crds_ctx = str(
        crds_ctx_hdr or getattr(dm.meta.ref_file.crds, 'context_used', '')
    )

    # Full time axis from Stage-4 LCData (rounded and offset)
    if 'time' not in lc.coords and 'time' not in lc:
        raise KeyError("Stage-4 LCData must contain a 'time' coordinate.")
    lc_time = lc.coords.get('time', lc['time']).values
    full_time = np.round(lc_time + timeOffset, timeDecimals)

    # Stage-5 fit time (subset) aligned to full_time
    fit_time = np.round(fit['time'].values + timeOffset, timeDecimals)

    def _to_full(values):
        s = pd.Series(np.asarray(values), index=fit_time)
        s = s[~s.index.duplicated(keep='first')]
        return s.reindex(full_time).values

    # ---- Series on full_time ----
    v = int(visit)

    # RAW FLUX from Stage-4 'data' (flatten), scaled by Stage-5 divisor
    lc_raw_full = np.ravel(lc['data'].values).astype(float)

    # Overlap indices and multiplicative divisor c: y = x / c
    _, i_full, i_fit = np.intersect1d(
        full_time, fit_time, assume_unique=False, return_indices=True
    )
    x = lc_raw_full[i_full].astype(float)
    y = fit['lcdata'].values[i_fit].astype(float)
    c = np.dot(x, x) / np.dot(x, y)  # Stage-5 divisor (assumed > 0)
    raw_full_norm = lc_raw_full / c

    raw_flux = _series1(
        np.round(raw_full_norm, lightcurveDecimals),
        full_time, v, units=lightcurve_units, decimals=None,
    )
    raw_flux.name = 'rawFlux'

    # RAW FLUX ERROR from Stage-4 'err', divided by the same c
    lc_err_full = np.ravel(lc['err'].values).astype(float)
    raw_err_norm = lc_err_full / c
    raw_flux_err = _series1(
        np.round(raw_err_norm, lightcurveDecimals),
        full_time, v, units=lightcurve_units, decimals=None,
    )
    raw_flux_err.name = 'rawFluxErr'

    # Stage-5 models reindexed to full_time (NaN where dropped)
    astro = _series1(
        _to_full(fit['astrophysical model'].values),
        full_time, v, units=lightcurve_units, decimals=lightcurveDecimals,
    )
    astro.name = 'astroModel'

    if 'GP' in fit.keys():
        GP = fit['GP'].values
    else:
        GP = np.zeros_like(fit['model'].values)
    noise_vals = fit['model'].values/fit['astrophysical model'].values + GP
    noise = _series1(
        _to_full(noise_vals),
        full_time, v, units=lightcurve_units, decimals=lightcurveDecimals,
    )
    noise.name = 'noiseModel'

    full_vals = fit['model'].values + GP
    full = _series1(
        _to_full(full_vals),
        full_time, v, units=lightcurve_units, decimals=lightcurveDecimals,
    )
    full.name = 'fullModel'

    cleaned_vals = (
        fit['astrophysical model'].values + (fit['residuals'].values - GP)
    )
    cleaned = _series1(
        _to_full(cleaned_vals),
        full_time, v, units=lightcurve_units, decimals=lightcurveDecimals,
    )
    cleaned.name = 'cleanedFlux'

    # Centroid diagnostics from Stage-4 (on the full time axis)
    x = lc['centroid_x'].values + np.nanmin(spec_ds.x.values)
    y = lc['centroid_y'].values + np.nanmin(spec_ds.y.values)
    cenx = _series1(x, full_time, v, units='pix')
    ceny = _series1(y, full_time, v, units='pix')
    censx = _series1(lc['centroid_sx'].values, full_time, v, units='pix')
    censy = _series1(lc['centroid_sy'].values, full_time, v, units='pix')

    cenx.name, ceny.name = 'centroid_x', 'centroid_y'
    censx.name, censy.name = 'centroid_sx', 'centroid_sy'

    # Assemble dataset
    ds = xr.Dataset(
        {
            'rawFlux': raw_flux,
            'rawFluxErr': raw_flux_err,
            'astroModel': astro,
            'noiseModel': noise,
            'fullModel': full,
            'cleanedFlux': cleaned,
            'centroid_x': cenx,
            'centroid_y': ceny,
            'centroid_sx': censx,
            'centroid_sy': censy,
        }
    )

    # Set coordinates explicitly
    ds = ds.assign_coords(visit=[v])
    ds = ds.assign_coords(time=full_time)
    ds.time.attrs['units'] = time_units

    # Per-visit metadata as columns (1× strings/floats)
    ds['SRC_DOI'] = _da1(str(SRC_DOI), v)
    ds['VISIT_ID'] = _da1(visit_id_str, v)
    ds['FILE_ID'] = _da1(file_id_str, v)
    ds['MJD_BEG'] = _da1(mjd_beg, v)
    ds['MJD_MID'] = _da1(mjd_mid, v)
    ds['MJD_END'] = _da1(mjd_end, v)
    ds['XPOSURE'] = _da1(xposure, v)
    ds['CAL_VER'] = _da1(cal_ver, v)
    ds['CRDS_VER'] = _da1(crds_ver, v)
    ds['CRDS_CTX'] = _da1(crds_ctx, v)
    ds['PIPELINE'] = _da1(pipe_line, v)
    ds['PIPE_VER'] = _da1(pipe_ver, v)
    ds['filter'] = _da1(str(filter_name), v)
    ds['subarray'] = _da1(str(subarray), v)
    ds['date_obs'] = _da1(date_obs, v)

    # Optional: include R_*/S_* cards for provenance completeness
    for key, val in r_from_fits.items():
        ds[key] = _da1(str(val), v)
    for key, val in s_from_fits.items():
        ds[key] = _da1(str(val), v)

    # Global invariants
    ds.attrs.update(
        {
            'HLSPVER': HLSPVER,
            'HLSPID': HLSPID,
            'HLSPNAME': HLSPNAME,
            'HLSP_PI': HLSP_PI,
            'HLSPLEAD': HLSPLEAD,
            'DOI': DOI,
            'STAR': STAR,
            'PLANET': PLANET,
            'HLSPTARG': PLANET,
            'OBSERVAT': observat,
            'TELESCOP': telescop,
            'INSTRUME': instrume,
            'RADESYS': RADESYS,
            'TIMESYS': TIMESYS,
            'LICENSE': LICENSE,
            'LICENURL': LICENURL,
            'RA_TARG': ra_targ,
            'DEC_TARG': dec_targ,
            'PROPOSID': proposid,
        }
    )

    return ds


def save_lightcurve_hdf5(ds, out_dir='.'):
    """
    Save the light-curve dataset to an HDF5-based netCDF file.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset produced by :func:`build_lightcurve_dataset`.
    out_dir : str, optional
        Directory where the file will be written. Defaults to current dir.

    Returns
    -------
    out_path : str
        Path to the written file on disk.
    """
    instrume = ds.attrs['INSTRUME']
    planet = ds.attrs['PLANET']
    visit = int(np.asarray(ds['visit'].values).item())
    filt = str(ds['filter'].values[0])
    hlspver = ds.attrs['HLSPVER']

    planet_fn = ''.join(planet.lower().split())
    out_name = (
        f'hlsp_rocky-worlds_jwst_{instrume.lower()}_{planet_fn}-'
        f'obs{visit:03d}_{filt.lower()}_v{hlspver.lower()}_lc.h5'
    )
    out_path = str(Path(out_dir) / out_name)

    # Compress floating arrays
    enc = {
        name: {'zlib': True, 'complevel': 4}
        for name in ds.data_vars
        if np.issubdtype(ds[name].dtype, np.floating)
    }
    ds.to_netcdf(
        out_path,
        engine='h5netcdf',
        invalid_netcdf=True,
        encoding=enc,
        mode='w',
    )
    return out_path


def save_lightcurve_multi_hdf5(datasets, checkpoint, out_dir='.', hlspver=None):
    """
    Save multiple single-visit light-curve datasets into one HDF5 file,
    one HDF5 group per visit (no NaN padding across disjoint time axes).

    Parameters
    ----------
    datasets : list of xarray.Dataset
        Each item is a single-visit light-curve dataset produced by
        :func:`build_lightcurve_dataset` with dims (visit, time).
    checkpoint : int
        Integer checkpoint number used in the filename, written as
        ``checkpoint##`` (two digits).
    out_dir : str, optional
        Directory where the file is written. Defaults to current directory.
    hlspver : str or None, optional
        HLSP version for the filename. If None, uses the first dataset's
        ``HLSPVER`` attribute.

    Returns
    -------
    out_path : str
        Path to the written file.

    Notes
    -----
    - Each visit is written to a separate group: ``/visit_###``.
    - This avoids creating a rectangular (visit, time) block and therefore
      avoids NaN padding for non-overlapping time axes.
    - Float arrays are compressed (zlib level 4).
    """
    if not datasets:
        raise ValueError("No datasets provided.")

    # Basic invariants from the first dataset
    first = datasets[0]
    instrume = first.attrs['INSTRUME']
    planet = first.attrs['PLANET']
    ver = hlspver or first.attrs.get('HLSPVER', '')
    if not ver:
        raise ValueError("HLSP version missing; pass hlspver or set attrs.")

    # Filename with checkpoint##
    planet_fn = ''.join(planet.lower().split())
    ckpt = f"checkpoint{int(checkpoint):02d}"
    out_name = (
        f"hlsp_rocky-worlds_jwst_{instrume.lower()}_{planet_fn}-"
        f"{ckpt}_v{ver.lower()}_lc.h5"
    )
    out_path = str(Path(out_dir) / out_name)

    # Write each visit into its own HDF5 group
    mode = 'w'
    for ds in datasets:
        # Validate shape
        if 'visit' not in ds.coords or 'time' not in ds.dims:
            raise ValueError("Each dataset must have 'visit' coord and 'time'.")
        if ds.dims.get('visit', 0) != 1:
            raise ValueError("Each dataset must be single-visit (visit=1).")

        visit_idx = int(np.asarray(ds['visit'].values).item())
        group = f"visit_{visit_idx:03d}"

        # Compression for floats
        enc = {
            name: {'zlib': True, 'complevel': 4}
            for name in ds.data_vars
            if np.issubdtype(ds[name].dtype, np.floating)
        }

        ds.to_netcdf(
            out_path,
            engine='h5netcdf',
            invalid_netcdf=True,
            encoding=enc,
            mode=mode,          # 'w' for first write, then 'a'
            group=group,
        )
        mode = 'a'

    return out_path

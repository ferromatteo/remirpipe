#!/usr/bin/env python3
"""
REMIR Pipeline
Complete processing pipeline with all parameters in config.yaml
"""

# Standard library imports
import argparse
import gzip
import heapq
import itertools
import logging
import os
import shutil
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime
from io import StringIO

# Third-party imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for batch processing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import sep
import yaml
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points
from photutils.aperture import CircularAperture, aperture_photometry
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.spatial import cKDTree
import numpy.lib.recfunctions as rfn
from scipy.spatial.distance import cdist

# Global statistics to collect pipeline-wide counts
STATS = defaultdict(int)

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION LOADER
# ============================================================================

def load_config(config_path):
    """Load configuration from YAML file."""
    if config_path is None:
        # Default: look in script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'config.yaml')
    elif not os.path.isabs(config_path):
        # User specified: use relative to current directory
        config_path = os.path.abspath(config_path)
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='REMIR Pipeline - Complete reduction and analysis pipeline'
    )
    parser.add_argument('-i', '--input', required=True,
                        help='Input directory with FITS files')
    parser.add_argument('-o', '--output', default=None,
                        help='Output directory (default: same as input)')
    parser.add_argument('-c', '--config', default=None,
                        help='Configuration file (default: config.yaml in script directory)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('-d', '--delete-tmp', action='store_true',
                        help='Delete tmp directory at the end')
    parser.add_argument('-s', '--scale-constraint', action='store_true',
                        help='Apply scale constraints (0.95-1.05) for astrometry')
    parser.add_argument('-co', '--clean-output', action='store_true',
                        help='Clean existing output directories before starting')
    parser.add_argument('-t', '--target', nargs='+', default=None,
                        help='Target OBJECT name(s) for astrometry/photometry only (e.g. -t NGC1234 M31). '
                             'All data are processed but astrometry and photometry are performed only on matching OBJECTs.')
    parser.add_argument('-f', '--target-file', default=None,
                        help='Text file with target positions (RA DEC radius_arcsec per line). '
                             'Supports sexagesimal (HH:MM:SS.ss DD:MM:SS.ss) or decimal degrees.')
    
    return parser.parse_args()


def parse_target_file(filepath):
    """Parse a target position file.
    
    Each line: RA DEC [radius_arcsec]
    Supported formats:
        Sexagesimal:  13:58:09.72 -64:44:05.26 1.5
        Decimal deg:  209.540 -64.735 2.0
        No radius:    209.540 -64.735          (defaults to 2.0 arcsec)
    
    Returns list of dicts with 'ra_deg', 'dec_deg', 'radius_arcsec'.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    
    DEFAULT_RADIUS_ARCSEC = 2.0
    
    targets = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            ra_str, dec_str = parts[0], parts[1]
            radius = float(parts[2]) if len(parts) >= 3 else DEFAULT_RADIUS_ARCSEC
            
            if ':' in ra_str:
                coord = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
            else:
                coord = SkyCoord(float(ra_str), float(dec_str), unit=u.deg)
            
            targets.append({
                'ra_deg': coord.ra.deg,
                'dec_deg': coord.dec.deg,
                'radius_arcsec': radius
            })
    
    return targets


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_file, verbose):
    """Setup logging to pipelog.txt file and console."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO if verbose else logging.WARNING)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter('%(message)s')

    # File handler (pipelog.txt only)
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO if verbose else logging.WARNING)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


# --------------------------------------------------------------------------
# Logging helpers for clearer, structured terminal output
# --------------------------------------------------------------------------
def log_big_divider(title: str):
    """Print a big divider block for major sections (per-coadd).

    This is intentionally plain-text so it appears clearly both in console
    and in log files.
    """
    logging.info("")
    logging.info("=" * 80)
    logging.info(f"= {title}")
    logging.info("=" * 80)


def log_small_divider():
    """Print a small divider for individual attempts."""
    logging.info("-" * 60)


def log_attempt_start(npix, thresh, filt, is_reflection=False, idx=None, total=None):
    log_small_divider()
    refl = " (reflection)" if is_reflection else ""
    prefix = f"[{idx}/{total}] " if idx is not None and total is not None else ""
    logging.info(f"{prefix}Attempt: npix={npix}, thresh={thresh}, filter={filt}{refl}")


def log_attempt_result(success, npix, thresh, filt, is_reflection=False, reason=None):
    refl = " (reflection)" if is_reflection else ""
    sym = "✓" if success else "✗"
    logging.info(f"{sym} Attempt finished: npix={npix}, thresh={thresh}, filter={filt}{refl} -> {'SUCCESS' if success else 'FAILED'}: {reason}")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def determine_proctype(header):
    """
    Determine PROCTYPE from FITS header.
    
    Returns:
    --------
    0 = FLAT (OBSTYPE = FLATF)
    1 = STD (OBSTYPE = STDSTAR)
    2 = SCI (IMAGETYP = OBJECT and OBSTYPE not in [FLATF, STDSTAR])
    -1 = FOCUS (IMAGETYP = FOCUS)
    None = Unknown
    """
    obstype = header.get('OBSTYPE', '').strip().upper()
    imagetyp = header.get('IMAGETYP', '').strip().upper()
    
    if obstype == 'FLATF':
        return 0
    elif obstype == 'STDSTAR':
        return 1
    elif imagetyp == 'OBJECT' and obstype not in ['FLATF', 'STDSTAR']:
        return 2
    elif imagetyp == 'FOCUS':
        return -1
    else:
        return None

def get_central_region(data, fraction=0.8):
    """Get central region of array."""
    h, w = data.shape
    margin_h = int(h * (1 - fraction) / 2)
    margin_w = int(w * (1 - fraction) / 2)
    return data[margin_h:h-margin_h, margin_w:w-margin_w]

def sigma_clipped_median(data, sigma=3.0):
    """Calculate sigma-clipped median."""
    _, median, _ = sigma_clipped_stats(data, sigma=sigma)
    return median

def format_date_like_dateobs(date_obs_str):
    """Format current date to match DATE-OBS format."""
    # Parse DATE-OBS to get format
    try:
        # Assume ISO format like '2024-12-16T12:34:56.123'
        dt = datetime.now()
        # Match precision of DATE-OBS
        if '.' in date_obs_str:
            # Has fractional seconds
            precision = len(date_obs_str.split('.')[-1])
            return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-6+precision]
        else:
            return dt.strftime('%Y-%m-%dT%H:%M:%S')
    except:
        return datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]

def average_date_obs(date_obs_list):
    """Calculate average DATE-OBS."""
    times = [Time(d, format='isot') for d in date_obs_list]
    avg_mjd = np.mean([t.mjd for t in times])
    avg_time = Time(avg_mjd, format='mjd')
    
    # Match format of first date
    if '.' in date_obs_list[0]:
        precision = len(date_obs_list[0].split('.')[-1])
        return avg_time.isot[:19+1+precision]
    else:
        return avg_time.isot[:19]


def load_calibration_files(data_folder, filt, dither_angle, config, verbose=False):
    """Load flat field calibration file.
    
    Args:
        data_folder: Directory containing calibration files
        filt: Filter name (J, H, K)
        dither_angle: Dither angle from header
        config: Pipeline configuration dictionary
        verbose: Enable verbose logging
    
    Returns:
        dict with 'flat' or None if file not found
    """
    dither_str = f"{int(round(dither_angle))}"
    flat_file = os.path.join(data_folder, f"{filt}_dither{dither_str}_flat.fits")
    
    if not os.path.exists(flat_file):
        if verbose:
            logging.info(f"Flat calibration file not found: {flat_file}")
        return None
    
    # Load flat
    with fits.open(flat_file) as hdul:
        flat = hdul[0].data.astype(np.float32)
    
    if verbose:
        logging.info(f"Loaded calibration: {flat_file}")
    
    return {'flat': flat}


def calculate_rms_quality(rms, config):
    """Calculate RMS quality classification.
    
    Parameters:
        rms: RMS value in magnitudes
        config: Configuration dictionary
    
    Returns:
        str: Quality classification (VERY GOOD, GOOD, MEDIUM, POOR, VERY POOR, UNKNOWN)
    """
    rms_thresh = config['photometry']['quality_thresholds']['rms']
    
    if rms is None or not np.isfinite(rms) or rms < 0:
        return "UNKNOWN"
    elif rms < rms_thresh['very_good']:
        return "VERY GOOD"
    elif rms < rms_thresh['good']:
        return "GOOD"
    elif rms < rms_thresh['medium']:
        return "MEDIUM"
    elif rms < rms_thresh['poor']:
        return "POOR"
    else:
        return "VERY POOR"


def calculate_rejection_quality(n_used, n_total, config):
    """Calculate rejection quality classification.
    
    Parameters:
        n_used: Number of stars used after rejection
        n_total: Total number of stars before rejection
        config: Configuration dictionary
    
    Returns:
        tuple: (quality_string, rejected_fraction)
    """
    rejection_thresh = config['photometry']['quality_thresholds']['rejection']
    
    if n_total is None or n_total <= 0:
        return "UNKNOWN", 0.0
    
    rejected_fraction = (n_total - n_used) / n_total
    
    if rejected_fraction >= rejection_thresh['poor']:
        return "VERY POOR", rejected_fraction
    elif rejected_fraction >= rejection_thresh['medium']:
        return "POOR", rejected_fraction
    elif rejected_fraction >= rejection_thresh['good']:
        return "MEDIUM", rejected_fraction
    else:
        return "GOOD", rejected_fraction


# ============================================================================
# STEP 1: FILE PREPARATION
# ============================================================================

def gunzip_files(input_dir, verbose=False):
    """Decompress all gzipped FITS files in directory.
    
    Scans input_dir for .fits.gz files and decompresses them in place,
    creating .fits files alongside the compressed versions.
    
    Args:
        input_dir: Directory containing .fits.gz files
        verbose: Enable verbose logging
    """
    if verbose:
        logging.info("Step 1: Gunzipping files...")

    # Find .fits.gz recursively to handle nested folders
    gz_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.endswith('.fits.gz'):
                gz_files.append(os.path.join(root, f))

    for gz_path in gz_files:
        gz_name = os.path.basename(gz_path)

        # Skip macOS resource-fork files
        if gz_name.startswith('._'):
            if verbose:
                logging.info(f"  Skipping resource-fork file: {gz_name}")
            continue

        # Quick magic check: gzip files start with 0x1f 0x8b
        try:
            with open(gz_path, 'rb') as fh:
                magic = fh.read(2)
        except Exception as e:
            logging.warning(f"  Cannot read {gz_path}: {e}")
            continue

        if magic != b"\x1f\x8b":
            if verbose:
                logging.info(f"  Skipping {gz_name}: not a gzip file (magic={magic!r})")
            continue

        fits_path = gz_path[:-3]
        if verbose:
            logging.info(f"  Decompressing {gz_name}...")

        try:
            with gzip.open(gz_path, 'rb') as f_in, open(fits_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        except Exception as e:
            logging.warning(f"  Failed to decompress {gz_path}: {e}")

# # # ASRTROMETRY FUNCTION, TO COPY SOMEWHERE ELSE SINCE THIS IS GOLD
def try_astrometry(data, header, catalog_df, config, scale_constraint, filter_name, verbose=False, _is_reflection=False, attempt_offset=0, total_attempts=None):
    """Attempt astrometric calibration using quad-matching algorithm.
    
    Tries multiple combinations of detection parameters (min_pixels, threshold)
    and catalog filters to find a valid astrometric solution. Uses geometric
    quad matching to identify star patterns between detected sources and
    catalog positions.
    
    The algorithm:
    1. Detect sources with SEP using given parameters
    2. Build geometric quads from brightest sources
    3. Match quads between detections and catalog
    4. Compute similarity transform from matched quads
    5. Verify transform quality (number of matches, RMS residual)
    6. If successful, fit WCS using matched star pairs
    
    Args:
        data: two-dimensional image array
        header: FITS header with initial WCS estimate
        catalog_df: DataFrame with 2MASS catalog (RAJ2000, DEJ2000, J, H, K)
        config: Pipeline configuration dictionary
        scale_constraint: If True, reject transforms outside scale bounds
        filter_name: Image filter (J, H, K) for catalog magnitude selection
        verbose: Enable verbose logging
        _is_reflection: Internal flag for reflection attempts
        attempt_offset: Internal counter offset for attempt numbering
        total_attempts: Internal total attempt count for logging
    
    Returns:
        tuple: (wcs, n_matches, attempt_info) where:
            - wcs: Fitted WCS object or None if failed
            - n_matches: Number of matched stars
            - attempt_info: Dict with attempt details or None
    """

    min_pixels_list = config['detection']['min_pixels']
    threshold_sigma_list = config['detection']['threshold_sigma']
    aperture_radius = config['detection']['aperture_radius']

    astrometry_cfg = config['astrometry']
    print_best_only = bool(astrometry_cfg.get('print_best_only', False))
    start_time = time.perf_counter()
    filter_fallback = astrometry_cfg['filter_fallback']
    filters_to_try = filter_fallback.get(filter_name, ['H'])

    initial_wcs = WCS(header, relax=True)

    default_mag = astrometry_cfg['catalog']['default_mag']

    # Prepare attempt counting and numbering
    base_N = len(min_pixels_list) * len(threshold_sigma_list) * len(filters_to_try)
    # Only compute total_attempts if not passed from reflection call
    if total_attempts is None:
        total_attempts = base_N * (1 + int(astrometry_cfg.get('try_reflection', False)))

    attempt_idx = attempt_offset
    attempts_tried = 0
    attempts_success = 0
    attempts_failed = 0

    def finish_attempt(success, npix, thresh, filt, reason=None):
        nonlocal attempts_tried, attempts_success, attempts_failed
        attempts_tried += 1
        if success:
            attempts_success += 1
        else:
            attempts_failed += 1
        log_attempt_result(success, npix, thresh, filt, _is_reflection, reason)

    for npix in min_pixels_list:
        for thresh in threshold_sigma_list:
            for filt in filters_to_try:
                attempt_idx += 1
                log_attempt_start(npix, thresh, filt, _is_reflection, attempt_idx, total_attempts)

                try:
                    detections, data_sub, _ = detect_sources(data, config, npix, thresh)
                except Exception as e:
                    # Fatal error (e.g. SEP deblending overflow) - mark all remaining as failed and exit
                    logging.warning(f"    SEP error (skipping remaining attempts): {e}")
                    attempts_failed += (total_attempts - attempts_tried)
                    attempts_tried = total_attempts
                    # Jump to final summary (only if outermost call)
                    if not _is_reflection:
                        duration = time.perf_counter() - start_time
                        log_big_divider(f"Summary for {header.get('FILENAME', 'coadd')}")
                        logging.info(f"  Attempts tried: {attempts_tried}")
                        logging.info(f"  Successes: {attempts_success}")
                        logging.info(f"  Failures: {attempts_failed}")
                        logging.info(f"  Total runtime: {duration:.2f} s")
                        logging.info(f"  Skipped due to SEP error")
                        logging.info("All attempts done for this coadd -> Final: FAILED")
                    return None, 0, None
                # Quality check: minimum sources to even attempt astrometry
                min_sources = astrometry_cfg.get('min_sources', 3)
                if len(detections) < min_sources:
                    finish_attempt(False, npix, thresh, filt, f"<{min_sources} detections (image may be empty/bad)")
                    continue
                
                if len(detections) < 4:
                    finish_attempt(False, npix, thresh, filt, "<4 detections")
                    continue

                detections = photometry(detections, data_sub, aperture_radius)
                detections.sort(key=lambda s: s['mag'])

                n_det = int(astrometry_cfg.get('n_sources_detected', 25))
                n_cat = int(astrometry_cfg.get('n_sources_catalog', n_det))

                # Report detection counts
                logging.info(f"    Detection: {len(detections)} sources detected; using top {n_det}")

                det_sources_bright = detections[:n_det]
                if len(det_sources_bright) < 4:
                    finish_attempt(False, npix, thresh, filt, "<4 bright detections")
                    continue

                # Build quads using notebook-style tables and heap selection
                DET = np.array([[s['x'], s['y'], s['mag']] for s in det_sources_bright])

                det_table = Table([DET[:, 0], DET[:, 1], DET[:, 2]], names=('X', 'Y', 'MAG'))
                # Limit detection table to the configured number of detected sources
                det_table = det_table[:n_det]

                cat_filtered = catalog_df[catalog_df[filt] != default_mag].copy()
                if len(cat_filtered) < 4:
                    finish_attempt(False, npix, thresh, filt, "<4 catalog sources after filtering")
                    continue

                cat_sources = []
                cat_ra = cat_filtered['RAJ2000'].values
                cat_dec = cat_filtered['DEJ2000'].values
                cat_pix_x, cat_pix_y = initial_wcs.all_world2pix(cat_ra, cat_dec, 0)

                # Report catalog counts (rows, in-image, used)
                logging.info(f"    Catalog rows (filtered): {len(cat_filtered)}")

                margin_frac = config['detection'].get('margin_frac', 0.02)
                margin = int(margin_frac * max(data.shape))
                for i, (idx, row) in enumerate(cat_filtered.iterrows()):
                    x_pix = cat_pix_x[i]
                    y_pix = cat_pix_y[i]
                    if not np.isfinite(x_pix) or not np.isfinite(y_pix):
                        continue
                    if (-margin <= x_pix < data.shape[1] + margin and
                        -margin <= y_pix < data.shape[0] + margin):
                        cat_sources.append({
                            'x': float(x_pix),
                            'y': float(y_pix),
                            'mag': float(row[filt]),
                            'ra': float(row['RAJ2000']),
                            'dec': float(row['DEJ2000'])
                        })

                if len(cat_sources) < 4:
                    finish_attempt(False, npix, thresh, filt, "<4 catalog pixel sources")
                    continue

                logging.info(f"    Catalog in-image sources: {len(cat_sources)}; using top {n_cat}")

                cat_sources.sort(key=lambda s: s['mag'])
                # Use `n_sources_catalog` (falls back to n_det) for catalog-side quads
                cat_sources_bright = cat_sources[:n_cat]
                if len(cat_sources_bright) < 4:
                    finish_attempt(False, npix, thresh, filt, "<4 bright catalog sources")
                    continue

                CAT = np.array([[s['x'], s['y'], s['mag']] for s in cat_sources_bright])
                cat_table = Table([CAT[:, 0], CAT[:, 1], CAT[:, 2]], names=('X', 'Y', 'MAG'))

                if not print_best_only:
                    logging.info(f"    Building quads from {len(det_table)} det and {len(cat_table)} cat sources...")
                det_quads = build_quads_heap(det_table, G=astrometry_cfg['num_quads'])
                cat_quads = build_quads_heap(cat_table, G=astrometry_cfg['num_quads'])
                if not print_best_only:
                    logging.info(f"    Built {len(det_quads)} det quads, {len(cat_quads)} cat quads")

                matches = match_quads(det_quads, cat_quads, config)
                if not matches:
                    logging.info("    No quad matches found")
                    finish_attempt(False, npix, thresh, filt, "no quad matches")
                    continue

                cfg_top = int(astrometry_cfg.get('top_matches', 50))
                if cfg_top <= 0:
                    top_matches = len(matches)
                else:
                    top_matches = min(cfg_top, len(matches))
                pix_tol = astrometry_cfg['pix_tol']
                min_matches_rank = int(astrometry_cfg.get('min_matches_rank', 0))
                scale_min = float(astrometry_cfg.get('scale_min', 0.95))
                scale_max = float(astrometry_cfg.get('scale_max', 1.05))

                # For building quads we use the top-N bright sources, but for
                # evaluating candidate transforms we use *all* available detections
                # and all in-image catalog sources to get a robust match count.
                det_coords_all = np.array([[s['x'], s['y']] for s in detections])
                cat_coords_all = np.array([[s['x'], s['y']] for s in cat_sources])
                logging.info(f"    Matching will use {len(det_coords_all)} detections and {len(cat_coords_all)} catalog sources (full lists)")

                all_transformations = []
                attempted_transforms = []

                if not print_best_only:
                    logging.info(f"    Evaluating {top_matches} quad matches for consensus...")

                for match in matches[:top_matches]:
                    try:
                        cat_quad_coords = np.array(match['cat_quad']['coords'])
                        det_quad_coords = np.array(match['det_quad']['coords'])
                        scale, R, t = compute_similarity_transform(cat_quad_coords, det_quad_coords)
                    except Exception:
                        continue
                    # If -s was specified, enforce scale bounds immediately
                    if scale_constraint and not (scale_min <= scale <= scale_max):
                        if not print_best_only:
                            logging.info(f"      Skipping transform: scale={scale:.4f} outside [{scale_min},{scale_max}]")
                        continue

                    # Apply transform to all catalog coords and count matches
                    cat_transformed = scale * (cat_coords_all @ R.T) + t
                    tree = cKDTree(det_coords_all)
                    distances, indices = tree.query(cat_transformed, k=1)
                    matches_mask = distances < pix_tol
                    n_star_matches = int(np.sum(matches_mask))

                    # Record attempted transform (even if skipped) for diagnostics
                    matched_pairs = [(i, indices[i]) for i in range(len(cat_coords_all)) if matches_mask[i]]
                    if matched_pairs:
                        residuals = [np.linalg.norm(cat_transformed[i] - det_coords_all[j]) for i, j in matched_pairs]
                        rms = float(np.sqrt(np.mean(np.square(residuals))))
                    else:
                        rms = 999.0

                    attempted_transforms.append({
                        'scale': scale,
                        'angle_deg': np.rad2deg(np.arctan2(R[1, 0], R[0, 0])),
                        'tx': t[0],
                        'ty': t[1],
                        'n_matches': n_star_matches,
                        'rms': rms,
                    })

                    # Always skip transforms with zero matches
                    if n_star_matches == 0:
                        if not print_best_only:
                            logging.info("      Skipping transform: matches=0 (no matches)")
                        continue

                    # Skip transforms with too few matches (do not include in ranking)
                    if n_star_matches < min_matches_rank:
                        if not print_best_only:
                            logging.info(f"      Skipping transform: matches={n_star_matches} < min_matches_rank={min_matches_rank}")
                        continue

                    # Collect matched pairs and RMS (already computed above)
                    angle_deg = np.rad2deg(np.arctan2(R[1, 0], R[0, 0]))

                    # Only include transforms that meet the ranking threshold
                    if n_star_matches >= min_matches_rank:
                        all_transformations.append({
                            'scale': scale,
                            'angle_deg': angle_deg,
                            'tx': t[0],
                            'ty': t[1],
                            'R': R,
                            't': t,
                            'n_matches': n_star_matches,
                            'rms': rms,
                            'matched_pairs': matched_pairs
                        })

                if not all_transformations:
                    logging.info("    No valid transformations found")
                    # Print the best attempted transform (nearest candidate) for diagnostics
                    if attempted_transforms:
                        best = max(attempted_transforms, key=lambda t: (t['n_matches'], -t['rms']))
                        logging.info(
                            f"    Nearest candidate -> matches={best['n_matches']}, rms={best['rms']:.3f} px, scale={best['scale']:.4f}, angle={best['angle_deg']:.2f}°, tx={best['tx']:.2f}, ty={best['ty']:.2f}"
                        )
                    finish_attempt(False, npix, thresh, filt, "no valid transformations")
                    continue

                consensus_cfg = astrometry_cfg['consensus']

                # Filter out any transformations with zero matches as a safety
                all_transformations = [t for t in all_transformations if t['n_matches'] > 0]

                groups = defaultdict(list)

                # Bin-based grouping for consensus
                scale_bin = consensus_cfg.get('scale_bin', 0.02)
                angle_bin = consensus_cfg.get('angle_bin', 0.5)
                trans_bin = consensus_cfg.get('translation_bin', 5)

                for trans in all_transformations:
                    s_key = float(int(trans['scale'] / scale_bin) * scale_bin)
                    a_key = float(int(trans['angle_deg'] / angle_bin) * angle_bin)
                    tx_key = float(int(trans['tx'] / trans_bin) * trans_bin)
                    ty_key = float(int(trans['ty'] / trans_bin) * trans_bin)

                    group_key = (s_key, a_key, tx_key, ty_key)
                    groups[group_key].append(trans)

                # Optionally merge nearby groups according to thresholds
                merge_translation_px = float(consensus_cfg.get('merge_translation_px', 0.0))
                merge_angle_deg = float(consensus_cfg.get('merge_angle_deg', 0.0))
                merge_scale = float(consensus_cfg.get('merge_scale', 0.0))

                if (merge_translation_px > 0) or (merge_angle_deg > 0) or (merge_scale > 0):
                    # Build list of group medians
                    group_keys = list(groups.keys())
                    n_groups = len(group_keys)
                    medians = []
                    for key in group_keys:
                        grp = groups[key]
                        med_s = float(np.median([t['scale'] for t in grp]))
                        med_a = float(np.median([t['angle_deg'] for t in grp]))
                        med_tx = float(np.median([t['tx'] for t in grp]))
                        med_ty = float(np.median([t['ty'] for t in grp]))
                        medians.append((med_s, med_a, med_tx, med_ty))

                    # Union-find for merging
                    parent = list(range(n_groups))

                    def find(x):
                        while parent[x] != x:
                            parent[x] = parent[parent[x]]
                            x = parent[x]
                        return x

                    def union(x, y):
                        rx, ry = find(x), find(y)
                        if rx != ry:
                            parent[ry] = rx

                    for i in range(n_groups):
                        for j in range(i + 1, n_groups):
                            s1, a1, tx1, ty1 = medians[i]
                            s2, a2, tx2, ty2 = medians[j]
                            # Angle difference: consider wrap-around
                            da = abs(a1 - a2) % 360.0
                            da = min(da, 360.0 - da)
                            dtrans = np.hypot(tx1 - tx2, ty1 - ty2)
                            if (abs(s1 - s2) <= merge_scale) and (da <= merge_angle_deg) and (dtrans <= merge_translation_px):
                                union(i, j)

                    clusters = defaultdict(list)
                    for idx in range(n_groups):
                        clusters[find(idx)].append(group_keys[idx])

                    if len(clusters) != len(group_keys):
                        # Build new merged groups
                        merged = defaultdict(list)
                        for rep, member_keys in clusters.items():
                            for k in member_keys:
                                merged_key = k  # temporary, will recompute median-based key
                                merged[merged_key].extend(groups[k])

                        # Recompute keys for merged groups using same binning/rounding rules
                        new_groups = defaultdict(list)
                        for old_key, trans_list in merged.items():
                            med_s = float(np.median([t['scale'] for t in trans_list]))
                            med_a = float(np.median([t['angle_deg'] for t in trans_list]))
                            med_tx = float(np.median([t['tx'] for t in trans_list]))
                            med_ty = float(np.median([t['ty'] for t in trans_list]))

                            if scale_bin:
                                s_key = float(int(med_s / scale_bin) * scale_bin)
                            else:
                                s_key = round(med_s, 4)

                            if angle_bin:
                                a_key = float(int(med_a / angle_bin) * angle_bin)
                            else:
                                a_key = round(med_a, 2)

                            if trans_bin:
                                tx_key = float(int(med_tx / trans_bin) * trans_bin)
                                ty_key = float(int(med_ty / trans_bin) * trans_bin)
                            else:
                                tx_key = round(med_tx, 1)
                                ty_key = round(med_ty, 1)

                            new_key = (s_key, a_key, tx_key, ty_key)
                            new_groups[new_key] = trans_list

                        groups = new_groups
                        logging.info(f"    Merged groups: {len(group_keys)} -> {len(groups)} using thresholds (tx={merge_translation_px}px, da={merge_angle_deg}°, ds={merge_scale})")

                # If no groups remain after filtering, report and move to next attempt
                if not groups:
                    logging.info("Grouped into 0 distinct groups")
                    logging.info("    No valid transformations found (after filtering)")
                    finish_attempt(False, npix, thresh, filt, "no valid transformations")
                    continue

                # Rank groups by a score combining median matches and median RMS
                # score = median_matches - rank_rms_weight * median_rms
                rank_rms_weight = consensus_cfg.get('rank_rms_weight', 1.0)
                scores = {}
                for key, group in groups.items():
                    med_matches = float(np.median([t['n_matches'] for t in group]))
                    med_rms = float(np.median([t['rms'] for t in group]))
                    score = med_matches - rank_rms_weight * med_rms
                    scores[key] = (score, len(group), med_matches, med_rms)

                # If print_best_only is set, show a compact summary; otherwise print full diagnostics
                if print_best_only:
                    # compute best group by score
                    best_key = max(scores.keys(), key=lambda k: (scores[k][0], scores[k][1]))
                    best_group = groups[best_key]
                    med_matches = int(np.median([t['n_matches'] for t in best_group]))
                    max_matches_best = int(np.max([t['n_matches'] for t in best_group]))
                    med_rms = float(np.median([t['rms'] for t in best_group]))
                    median_scale = float(np.median([t['scale'] for t in best_group]))
                    median_angle = float(np.median([t['angle_deg'] for t in best_group]))
                    median_tx = float(np.median([t['tx'] for t in best_group]))
                    median_ty = float(np.median([t['ty'] for t in best_group]))
                    # Will print summary after acceptance check below
                else:
                    logging.info(f"Grouped into {len(groups)} distinct groups")
                    logging.info("")
                
                # Choose best group by score (tie-breaker: larger group size)
                best_key = max(scores.keys(), key=lambda k: (scores[k][0], scores[k][1]))
                largest_group_key = best_key
                largest_group = groups[largest_group_key]

                # Only print detailed group info if not in compact mode
                if not print_best_only:
                    logging.info("============================================================")
                    logging.info(f"BEST GROUP (by score): {len(largest_group)} transformations")
                    logging.info("============================================================")
                    logging.info(f"Scale bin: {largest_group_key[0]:.2f}")
                    logging.info(f"Angle bin: {largest_group_key[1]:.2f}°")
                    logging.info(f"Translation bin: ({int(largest_group_key[2])}, {int(largest_group_key[3])}) px")
                    logging.info("")

                median_scale = float(np.median([t['scale'] for t in largest_group]))
                median_angle = float(np.median([t['angle_deg'] for t in largest_group]))
                median_tx = float(np.median([t['tx'] for t in largest_group]))
                median_ty = float(np.median([t['ty'] for t in largest_group]))
                median_n_matches = int(np.median([t['n_matches'] for t in largest_group]))
                max_n_matches = int(np.max([t['n_matches'] for t in largest_group]))
                median_rms = float(np.median([t['rms'] for t in largest_group]))

                # Decide acceptance based on min_matches (use MAX of group) and scale constraint
                accepted = True
                reject_reason = None
                # Fraction-based acceptance parameters
                min_match_fraction = float(astrometry_cfg.get('min_match_fraction', 0.25))
                accept_rms_px = float(astrometry_cfg.get('accept_rms_px', 1.5))
                n_det_total = len(det_coords_all) if 'det_coords_all' in locals() else 0
                match_fraction = (max_n_matches / n_det_total) if n_det_total > 0 else 0.0

                # Acceptance based on fraction of detected sources matched (preferred)
                if match_fraction >= min_match_fraction and median_rms <= accept_rms_px:
                    accepted = True
                    reject_reason = f"Accepted by fraction rule: frac={match_fraction:.2f} >= {min_match_fraction} and rms={median_rms:.3f} <= {accept_rms_px}"
                else:
                    accepted = False
                    reject_reason = f"Insufficient match fraction: frac={match_fraction:.2f} < {min_match_fraction} or rms={median_rms:.3f} > {accept_rms_px}"

                if accepted and scale_constraint and not (scale_min <= median_scale <= scale_max):
                    accepted = False
                    reject_reason = f"Scale {median_scale:.3f} outside [{scale_min}, {scale_max}]"

                # Compact output: print best candidate and whether accepted
                if print_best_only:
                    logging.info("=== BEST TRANSFORMATION (compact) ===")
                    logging.info(f"  Scale: {median_scale:.4f} | Angle: {median_angle:.2f}° | Translation: ({median_tx:.2f}, {median_ty:.2f}) px")
                    logging.info(f"  Star matches: {median_n_matches} (max {max_n_matches}) | RMS: {median_rms:.3f} px | Group size: {len(largest_group)}")
                    if accepted:
                        logging.info(f"✓ Best consensus transformation found: matches={max_n_matches}, rms={median_rms:.3f} px, scale={median_scale:.4f}, angle={median_angle:.2f}°, tx={median_tx:.2f}, ty={median_ty:.2f}")
                    else:
                        logging.info(f"✗ Best consensus candidate (rejected): matches={max_n_matches}, rms={median_rms:.3f} px, scale={median_scale:.4f}, angle={median_angle:.2f}°, tx={median_tx:.2f}, ty={median_ty:.2f}")
                        logging.info(f"    {reject_reason}")
                        finish_attempt(False, npix, thresh, filt, reject_reason)
                        continue
                else:
                    logging.info("MEDIAN PARAMETERS:")
                    logging.info(f"  Scale: {median_scale:.4f}")
                    logging.info(f"  Angle: {median_angle:.2f}°")
                    logging.info(f"  Translation: ({median_tx:.2f}, {median_ty:.2f}) px")
                    logging.info(f"  Star matches: {median_n_matches} (max {max_n_matches})")
                    logging.info(f"  RMS: {median_rms:.3f} px")
                    logging.info("")

                    logging.info("============================================================")
                    top_groups_display = int(astrometry_cfg.get('top_groups_display', 5))
                    logging.info(f"TOP {top_groups_display} GROUPS:")
                    logging.info("============================================================")
                    # Sort top groups by score (matches - weight * rms), tie-breaker is group size
                    sorted_groups = sorted(scores.items(), key=lambda item: (item[1][0], item[1][1]), reverse=True)
                    for idx, (key, (score, gsize, med_matches, med_rms)) in enumerate(sorted_groups[:top_groups_display], start=1):
                        logging.info(
                            f"{idx}. n={gsize:3d} | scale={key[0]:.2f}, angle={key[1]:6.2f}°, t=({int(key[2]):5d},{int(key[3]):5d}) | matches={int(med_matches)}, rms={med_rms:.3f}px | score={score:.3f}"
                        )
                    logging.info("")

                    # Check acceptance and report
                    if not accepted:
                        logging.info(f"    {reject_reason}")
                        finish_attempt(False, npix, thresh, filt, reject_reason or 'rejected')
                        continue
                    logging.info("✓ Using consensus transformation from largest group")
                    logging.info(f"    Success! {max_n_matches} matches (median {median_n_matches})")
                if scale_constraint:
                    scale_min = astrometry_cfg['scale_min']
                    scale_max = astrometry_cfg['scale_max']
                    if not (scale_min <= median_scale <= scale_max):
                        logging.info(
                            f"    Scale {median_scale:.3f} outside [{scale_min}, {scale_max}]"
                        )
                        continue

                angle_rad = np.deg2rad(median_angle)
                R_final = np.array([
                    [np.cos(angle_rad), -np.sin(angle_rad)],
                    [np.sin(angle_rad),  np.cos(angle_rad)]
                ])
                t_final = np.array([median_tx, median_ty])

                # Use DETECTED pixel positions for WCS fitting (not transformed catalog positions).
                # Transform catalog→detector to find matches, then use the actual detected coords.
                cat_all_coords = np.array([[s['x'], s['y']] for s in cat_sources])
                cat_all_transformed = median_scale * (cat_all_coords @ R_final.T) + t_final

                # Match transformed catalog positions to detected positions
                tree_det = cKDTree(det_coords_all)
                dists_wcs, idx_wcs = tree_det.query(cat_all_transformed, k=1)
                wcs_matched = dists_wcs < pix_tol

                # Ensure unique matches (one detection per catalog source)
                used_det = set()
                wcs_cat_indices = []
                wcs_det_indices = []
                # Sort by distance so closest matches win
                order = np.argsort(dists_wcs)
                for ci in order:
                    if not wcs_matched[ci]:
                        continue
                    di = idx_wcs[ci]
                    if di in used_det:
                        continue
                    used_det.add(di)
                    wcs_cat_indices.append(ci)
                    wcs_det_indices.append(di)

                # Build WCS from detected pixel positions + catalog sky coordinates
                wcs_xy = det_coords_all[wcs_det_indices]  # actual detected pixel positions
                wcs_sky = SkyCoord(
                    ra=[cat_sources[i]['ra'] for i in wcs_cat_indices] * u.deg,
                    dec=[cat_sources[i]['dec'] for i in wcs_cat_indices] * u.deg
                )



                try:
                    logging.info(f"Fitting WCS with {len(wcs_xy)} matched sources (detected pixels + catalog sky)...")
                    wcs = fit_wcs_from_points(
                        xy=wcs_xy.T,
                        world_coords=wcs_sky,
                        projection=astrometry_cfg.get('wcs_projection', 'TAN'),
                        sip_degree=astrometry_cfg.get('sip_degree', 3)
                    )
                    logging.info("✓ WCS fitted successfully!")
                    # Record which attempt succeeded
                    success_info = {
                        'npix': npix,
                        'thresh': thresh,
                        'filter': filt,
                        'is_reflection': _is_reflection,
                        'n_matches': max_n_matches
                    }
                    # Always print a per-attempt summary (include reflection if applicable)
                    finish_attempt(True, npix, thresh, filt, f"{median_n_matches} matches")
                    # Print per-coadd summary if outermost call
                    if not _is_reflection:
                        duration = time.perf_counter() - start_time
                        log_big_divider(f"Summary for {header.get('FILENAME', 'coadd')}")
                        logging.info(f"  Attempts tried: {attempts_tried}")
                        logging.info(f"  Successes: {attempts_success}")
                        logging.info(f"  Failures: {attempts_failed}")
                        logging.info(f"  Total runtime: {duration:.2f} s")
                    return wcs, median_n_matches, success_info
                except Exception as exc:
                    logging.info(f"    WCS fit failed: {exc}")
                    finish_attempt(False, npix, thresh, filt, 'wcs_fit_failed')
                    continue

    if astrometry_cfg['try_reflection'] and not _is_reflection:
        logging.info("    Trying reflection...")
        data_reflected = np.flip(data, axis=1)
        # Continue numbering when calling reflection by passing the current
        # attempt index as an offset so reflection attempts show the next
        # indices instead of restarting from 1. Also pass total_attempts so
        # [i/total] remains consistent.
        return try_astrometry(
            data_reflected,
            header,
            catalog_df,
            config,
            scale_constraint,
            filter_name,
            verbose=verbose,
            _is_reflection=True,
            attempt_offset=attempt_idx,
            total_attempts=total_attempts
        )

    # Print per-coadd summary if outermost call
    if not _is_reflection:
        duration = time.perf_counter() - start_time
        log_big_divider(f"Summary for {header.get('FILENAME', 'coadd')}")
        logging.info(f"  Attempts tried: {attempts_tried}")
        logging.info(f"  Successes: {attempts_success}")
        logging.info(f"  Failures: {attempts_failed}")
        logging.info(f"  Total runtime: {duration:.2f} s")
        logging.info("All attempts done for this coadd -> Final: FAILED")
    return None, 0, None


def filter_and_prepare_files(input_dir, config, verbose=False):
    """Filter and prepare FITS files for pipeline processing.
    
    Performs several preprocessing steps on each FITS file:
    1. Delete files with DITHID=98 or 99 (incomplete dither sequences)
    2. Fix invalid header values (e.g., NaN in WINDDIR)
    3. Add FILENAME keyword to header
    4. Determine and add PROCTYPE keyword (0=FLAT, 1=STD, 2=SCI, -1=FOCUS)
    5. Apply bad pixel mask if configured (marks masked pixels as NaN)
    
    Args:
        input_dir: Directory containing FITS files to prepare
        config: Pipeline configuration dictionary
        verbose: Enable verbose logging
    """
    if verbose:
        logging.info("Filtering and preparing files...")
    
    fits_files = [f for f in os.listdir(input_dir) if f.endswith('.fits')]
    
    # Load bad pixel mask if enabled and specified
    enable_mask = config.get('calibration', {}).get('enable_pixel_mask', True)
    mask_file = config.get('calibration', {}).get('mask_file')
    bad_pixel_mask = None
    if enable_mask and mask_file is not None:
        # Resolve path using data_folder from config
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_folder = config.get('paths', {}).get('data_folder', 'data')
        mask_path = os.path.join(script_dir, data_folder, mask_file)
        try:
            with fits.open(mask_path) as mask_hdul:
                bad_pixel_mask = mask_hdul[0].data
            if verbose:
                n_bad = np.sum(bad_pixel_mask == 0)
                logging.info(f"  Loaded bad pixel mask: {mask_path} ({n_bad} bad pixels)")
        except Exception as e:
            logging.warning(f"  Could not load mask file {mask_path}: {e}")
    
    for fits_file in fits_files:
        fits_path = os.path.join(input_dir, fits_file)
        
        with fits.open(fits_path, mode='update') as hdul:
            # Fix any invalid header cards first (e.g., WINDDIR = -nan)
            # Must be done before accessing header values
            hdul.verify('fix')
            
            header = hdul[0].header
            data = hdul[0].data
            
            # Check DITHID - delete if in exclusion list
            dithid = header.get('DITHID', 0)
            exclude_dithids = config['fits_markers']['dithid_exclude']
            if dithid in exclude_dithids:
                if verbose:
                    logging.info(f"  Deleting {fits_file} (DITHID={dithid})")
                STATS['files_deleted'] += 1
                # Close without verification to avoid errors from invalid header values
                hdul.close(output_verify='ignore')
                os.remove(fits_path)
                continue
            
            # Add/update FILENAME
            header['FILENAME'] = fits_file
            
            # Add/update PROCTYPE
            proctype = determine_proctype(header)
            header['PROCTYPE'] = proctype
            
            # Apply bad pixel mask
            if data is not None and len(data.shape) == 2:
                data = data.astype(np.float32)

                # Option 1: File-based mask (0=bad -> NaN)
                if bad_pixel_mask is not None:
                    if bad_pixel_mask.shape == data.shape:
                        mask_file_mask = (bad_pixel_mask == 0)
                        data[mask_file_mask] = np.nan
                    else:
                        logging.warning(f"  Mask file shape {bad_pixel_mask.shape} != image {data.shape}; skipping mask file for {fits_file}")



                hdul[0].data = data
            
            hdul.flush()


def classify_files(input_dir, tmp_dir, reduced_dir, config, verbose=False):
    """Classify files into old/new system directories based on DITHANGL keyword.
    
    Old system (pre-2025): Uses DWANGLE keyword for dither angle.
    New system (post-2025): Uses DITHANGL keyword for dither angle.
    
    FLAT and FOCUS files are moved directly to reduced_dir.
    Science files (SCI, STD) go to tmp/old or tmp/new for further processing.
    
    Args:
        input_dir: Source directory with prepared FITS files
        tmp_dir: Temporary directory for intermediate files
        reduced_dir: Output directory for final reduced files
        config: Pipeline configuration dictionary
        verbose: Enable verbose logging
    """
    if verbose:
        logging.info("Classifying files...")
    
    old_dir = os.path.join(tmp_dir, 'old')
    new_dir = os.path.join(tmp_dir, 'new')
    
    os.makedirs(old_dir, exist_ok=True)
    os.makedirs(new_dir, exist_ok=True)
    
    fits_files = [f for f in os.listdir(input_dir) if f.endswith('.fits')]
    
    for fits_file in fits_files:
        src_path = os.path.join(input_dir, fits_file)
        
        with fits.open(src_path) as hdul:
            header = hdul[0].header
            proctype = header.get('PROCTYPE', 2)
            
            # FOCUS files are not processed; copy straight to reduced (like flats)
            if proctype == -1:
                dst_path = os.path.join(reduced_dir, fits_file)
                shutil.copy2(src_path, dst_path)
                generate_preview_jpg(dst_path, config)
                continue
            
            # FLAT files go directly to reduced
            if proctype == 0:
                dst_path = os.path.join(reduced_dir, fits_file)
                shutil.copy2(src_path, dst_path)
                # Generate preview JPG for flat
                generate_preview_jpg(dst_path, config)
                continue
            
            # Check for DITHANGL keyword
            has_dithangl = 'DITHANGL' in header
            
            if has_dithangl:
                dst_dir = new_dir
            else:
                dst_dir = old_dir
            
            dst_path = os.path.join(dst_dir, fits_file)
            shutil.copy2(src_path, dst_path)
            
            # Update keywords in the copy
            with fits.open(dst_path, mode='update') as hdul_dst:
                hdul_dst[0].header['PROCSTAT'] = 0
                hdul_dst[0].header['PSTATSUB'] = 0
                hdul_dst[0].header['ASTROP'] = 0
                hdul_dst.flush()


def group_files(system_dir, config, verbose=False):
    """Group files by OBJECT/FILTER/OBSID/SUBID with time constraint."""
    if verbose:
        logging.info(f"Grouping files in {system_dir}...")
    
    fits_files = [f for f in os.listdir(system_dir) if f.endswith('.fits')]
    
    max_time_gap_hours = config['grouping']['max_time_gap_hours']
    
    # Collect file info
    file_infos = []
    for fits_file in fits_files:
        fits_path = os.path.join(system_dir, fits_file)
        
        with fits.open(fits_path) as hdul:
            header = hdul[0].header
            
            obj = header.get('OBJECT', 'UNKNOWN').strip()
            filt = header.get('FILTER', 'H').strip()
            obsid = header.get('OBSID', 0)
            subid = header.get('SUBID', 0)
            ndithers = header.get('NDITHERS', 1)
            date_obs = header.get('DATE-OBS', '')
            
            try:
                dt = Time(date_obs, format='isot').datetime
            except:
                dt = datetime.now()
            
            file_infos.append({
                'file': fits_file,
                'path': fits_path,
                'object': obj,
                'filter': filt,
                'obsid': obsid,
                'subid': subid,
                'ndithers': ndithers,
                'date_obs': date_obs,
                'datetime': dt,
                'header': header
            })
    
    # Sort by datetime
    file_infos.sort(key=lambda x: x['datetime'])
    
    # Group by OBJECT/FILTER/OBSID/SUBID
    groups = defaultdict(list)
    for fi in file_infos:
        key = (fi['object'], fi['filter'], fi['obsid'], fi['subid'])
        groups[key].append(fi)
    
    # Split groups by time gap
    final_groups = []
    for key, files in groups.items():
        if len(files) == 0:
            continue
        
        current_group = [files[0]]
        
        for i in range(1, len(files)):
            time_diff = (files[i]['datetime'] - files[i-1]['datetime']).total_seconds() / 3600
            
            if time_diff > max_time_gap_hours:
                # Start new group
                final_groups.append({
                    'key': key,
                    'files': current_group,
                    'ndithers': current_group[0]['ndithers']
                })
                current_group = [files[i]]
            else:
                current_group.append(files[i])
        
        if current_group:
            final_groups.append({
                'key': key,
                'files': current_group,
                'ndithers': current_group[0]['ndithers']
            })
    
    return final_groups


def validate_groups(groups, verbose=False):
    """Validate groups: complete, incomplete, or defective."""
    valid_groups = []
    incomplete_groups = []
    defective_groups = []
    
    for group in groups:
        files = group['files']
        ndithers = group['ndithers']
        n_files = len(files)
        
        # Check if all files have same NDITHERS
        ndithers_values = set(f['ndithers'] for f in files)
        
        if len(ndithers_values) > 1:
            # Defective: inconsistent NDITHERS
            if verbose:
                logging.warning(f"  Defective group {group['key']}: inconsistent NDITHERS {ndithers_values}")
            defective_groups.append(group)
            continue
        
        if n_files == ndithers:
            # Complete
            valid_groups.append(group)
        elif n_files > ndithers:
            # Defective: too many files
            if verbose:
                logging.warning(f"  Defective group {group['key']}: {n_files} files but NDITHERS={ndithers}")
            defective_groups.append(group)
        elif n_files < 3:
            # Incomplete with < 3 files: skip
            if verbose:
                logging.warning(f"  Skipping group {group['key']}: only {n_files} files")
            defective_groups.append(group)
        else:
            # Incomplete but processable
            if verbose:
                logging.info(f"  Incomplete group {group['key']}: {n_files}/{ndithers} files")
            incomplete_groups.append(group)
    
    return valid_groups, incomplete_groups, defective_groups


def process_group_sky_subtraction(group, config, output_dir, verbose=False):
    """Perform sky subtraction and flat field correction on a dither group.
    
    SIMPLIFIED WORKFLOW:
    1. Load raw masked frames (pixel mask already applied in file prep)
    2. Level normalization: compute median of central 80% for each frame,
       then scale all frames so medians equal the mean of all medians
    3. Create single sky from median of all N leveled frames
    4. Sky subtract and flat field: (raw - sky) / flat
    
    Noise propagation:
    - Input noise = sqrt(data_raw/gain + (read_noise/gain)^2)  [all in ADU]
    - After leveling: noise *= scale_factor (tracks data units)
    - Sky noise = median_factor * sqrt(sum(leveled_noise^2)) / N_eff
    - After sky subtraction: sqrt(leveled_noise^2 + sky_noise^2)
    - After flat division: final_noise / flat
    
    Args:
        group: Dict with 'files' list and 'key' tuple (object, filter, obsid, subid)
        config: Pipeline configuration dictionary
        output_dir: Directory to write sky-subtracted files
        verbose: Enable verbose logging
    
    Returns:
        tuple: (sky_path, skysub_files, group_name) where:
            - sky_path: Path to saved sky frame FITS
            - skysub_files: List of dicts with sky-subtracted file info
            - group_name: String identifier for the group
    """
    files = group['files']
    key = group['key']
    group_name = f"{key[0]}_{key[2]}_{key[3]}_{key[1]}"  # OBJECT_OBSID_SUBID_FILTER
    filt = key[1]  # Filter from group key
    
    if verbose:
        logging.info(f"  Processing group {group_name} ({len(files)} files)...")
    
    gain = config['detector']['gain']
    read_noise = config['detector']['read_noise']
    
    # ========== STEP 1: SETUP ==========
    enable_flat = config.get('calibration', {}).get('enable_flat_correction', True)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder_rel = config['paths'].get('data_folder', 'data')
    data_folder = os.path.join(script_dir, data_folder_rel)
    
    # Sky subtraction parameters
    sky_central_fraction = config['sky_subtraction']['central_fraction']
    sky_sigma_clip = config['sky_subtraction']['sigma_clip']
    
    # ========== STEP 2: LOAD RAW DATA (pixel mask already applied) ==========
    data_list = []
    noise_list = []
    headers = []
    
    for file_info in files:
        with fits.open(file_info['path']) as hdul:
            data_raw = hdul[0].data.astype(np.float32)
            header = hdul[0].header.copy()
        
        # Initial noise (all in ADU²)
        # Variance = data_raw/gain + (RON/gain)^2
        variance = data_raw / gain + (read_noise / gain)**2
        noise_initial = np.sqrt(variance)
        
        # Handle non-finite values
        noise_initial[~np.isfinite(data_raw)] = np.nan
        
        data_list.append(data_raw)
        noise_list.append(noise_initial)
        headers.append(header)
    
    # ========== STEP 3: LEVEL NORMALIZATION ==========
    # Compute median of central 80% for each frame
    medians = []
    for data in data_list:
        central = get_central_region(data, sky_central_fraction)
        med = sigma_clipped_median(central, sigma=sky_sigma_clip)
        medians.append(med)
    
    # Scale all frames so medians equal the mean of all medians
    mean_median = np.mean(medians)
    data_leveled = []
    noise_leveled = []
    for data, noise, med in zip(data_list, noise_list, medians):
        if med > 0:
            scale_factor = mean_median / med
            data_leveled.append(data * scale_factor)
            noise_leveled.append(noise * scale_factor)
        else:
            data_leveled.append(data.copy())
            noise_leveled.append(noise.copy())
    
    if verbose:
        logging.info(f"    Leveled {len(data_leveled)} frames to mean median={mean_median:.1f}")
    
    # ========== STEP 3b: BUILD SKY + POST-MEDIAN RESIDUAL CLEANUP ==========
    # The median of N dithered frames already rejects most stars (they land
    # on different pixels).  Only a few bright residual patches survive.
    # Strategy: build median sky first, then detect and replace only those
    # residual positive patches with realistic pixel values drawn from
    # nearby uncontaminated sky (matching local mean + noise statistics).
    
    sky_mask_cfg = config.get('sky_subtraction', {}).get('source_masking', {})
    sm_thresh = sky_mask_cfg.get('threshold_sigma', 2.0)
    sm_minarea = sky_mask_cfg.get('min_area', 3)
    sm_bw = sky_mask_cfg.get('bw', 64)
    sm_bh = sky_mask_cfg.get('bh', 64)
    
    # ========== STEP 4: CREATE SINGLE SKY FROM ALL N FRAMES ==========
    data_stack = np.array(data_leveled)
    sky_pattern = np.nanmedian(data_stack, axis=0)
    
    # Sky noise (using all N frames)
    noise_stack = np.array(noise_leveled)
    var_stack = np.nansum(noise_stack**2, axis=0)
    N_eff = np.sum(np.isfinite(data_stack), axis=0)
    N_eff[N_eff == 0] = 1
    noise_sky = config['sky_subtraction']['noise_median_factor'] * np.sqrt(var_stack) / N_eff
    
    # ========== STEP 4b: CLEAN RESIDUAL POSITIVE PATCHES IN SKY ==========
    # After the median, detect positive residual clusters (surviving star
    # flux) and replace them with values sampled from neighboring sky pixels
    # so the cleaned region has the same local mean and standard deviation.
    if sky_mask_cfg.get('enabled', False):
        try:
            sky_work = np.ascontiguousarray(
                np.nan_to_num(sky_pattern, nan=0.0).astype(np.float32))
            nan_mask_sky = ~np.isfinite(sky_pattern)
            bkg_sky = sep.Background(sky_work, bw=sm_bw, bh=sm_bh, mask=nan_mask_sky)
            sky_sub = sky_work - bkg_sky.back()
            sky_rms = bkg_sky.globalrms
            
            # Detect only the surviving positive residuals
            residual_objs = sep.extract(sky_sub, sm_thresh * sky_rms,
                                         minarea=sm_minarea, mask=nan_mask_sky)
            
            if len(residual_objs) > 0:
                # Build mask of contaminated pixels (tight: just the detected footprint)
                contam_mask = nan_mask_sky.copy()
                for obj in residual_objs:
                    try:
                        sep.mask_ellipse(contam_mask, obj['x'], obj['y'],
                                         obj['a'], obj['b'], obj['theta'],
                                         r=3.0)
                    except Exception:
                        pass
                
                # For each contaminated pixel, replace with a random draw
                # from nearby uncontaminated sky pixels (preserves noise texture)
                ny_s, nx_s = sky_pattern.shape
                clean_sky = sky_pattern.copy()
                contam_yx = np.argwhere(contam_mask & ~nan_mask_sky)
                
                if len(contam_yx) > 0:
                    # Build a pool of clean sky pixel values for local sampling
                    annulus_inner = 5
                    annulus_outer = 15
                    rng = np.random.default_rng(42)
                    
                    for cy, cx in contam_yx:
                        # Collect clean pixels in an annulus around this pixel
                        y_lo = max(0, cy - annulus_outer)
                        y_hi = min(ny_s, cy + annulus_outer + 1)
                        x_lo = max(0, cx - annulus_outer)
                        x_hi = min(nx_s, cx + annulus_outer + 1)
                        
                        local_patch = sky_pattern[y_lo:y_hi, x_lo:x_hi]
                        local_contam = contam_mask[y_lo:y_hi, x_lo:x_hi]
                        local_nan = nan_mask_sky[y_lo:y_hi, x_lo:x_hi]
                        
                        # Only use pixels outside inner radius and not contaminated
                        yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
                        dist2 = (yy - cy)**2 + (xx - cx)**2
                        in_annulus = (dist2 >= annulus_inner**2) & (dist2 <= annulus_outer**2)
                        good = in_annulus & ~local_contam & ~local_nan & np.isfinite(local_patch)
                        
                        pool = local_patch[good]
                        if len(pool) >= 5:
                            # Random draw from the pool — preserves local mean + noise
                            clean_sky[cy, cx] = rng.choice(pool)
                        else:
                            # Fallback: use local background estimate
                            clean_sky[cy, cx] = bkg_sky.back()[cy, cx]
                    
                    sky_pattern = clean_sky
                    
                    if verbose:
                        n_res = len(residual_objs)
                        pct = len(contam_yx) / sky_pattern.size * 100
                        logging.info(f"    Sky cleanup: {n_res} residual patches, "
                                     f"{len(contam_yx)} pixels ({pct:.2f}%) replaced")
        except Exception:
            pass  # keep original sky_pattern
    
    # Save sky pattern
    sky_filename = f"{group_name}_sky.fits"
    sky_path = os.path.join(output_dir, sky_filename)
    
    sky_header = headers[0].copy()
    sky_header['DITHID'] = config['fits_markers']['dithid_sky']
    sky_header['PROCSTAT'] = config['fits_markers']['procstat_reduced']
    sky_header['PSTATSUB'] = config['fits_markers']['pstatsub_sky']
    sky_header['FILENAME'] = sky_filename
    sky_header['DATE'] = format_date_like_dateobs(headers[0]['DATE-OBS'])
    sky_header['HISTORY'] = f"Sky from median of {len(files)} leveled frames"
    sky_header['HISTORY'] = f"Leveling: scaled to mean median={mean_median:.1f}"
    
    fits.writeto(sky_path, sky_pattern, sky_header, overwrite=True)
    STATS['sky_frames'] += 1
    
    if verbose:
        logging.info(f"    Created single sky pattern from {len(files)} frames")
    
    # ========== STEP 5: SKY SUBTRACTION + FLAT FIELD ==========
    # Check if LOO + Iterative Gaussian mode is enabled (supports per-filter override)
    loo_ig_cfg = config.get('sky_subtraction', {}).get('loo_iterative_gauss', {})
    use_loo_ig = loo_ig_cfg.get('enabled', False)
    
    # Per-filter override: filter_enable map takes precedence over global 'enabled'
    filter_enable = loo_ig_cfg.get('filter_enable', {})
    if filt in filter_enable:
        use_loo_ig = bool(filter_enable[filt])
    
    if verbose:
        logging.info(f"    LOO+IterGauss for filter {filt}: {'enabled' if use_loo_ig else 'disabled'}")
    
    if use_loo_ig:
        # LOO sky: for each frame, sky = median of the OTHER N-1 frames
        # Median naturally rejects dithered sources; residual cleanup
        # already applied to sky_pattern above via the same mechanism
        loo_skies = []
        for i in range(len(data_leveled)):
            others = [data_leveled[j] for j in range(len(data_leveled)) if j != i]
            loo_skies.append(np.nanmedian(np.array(others), axis=0))
        
        # LOO noise: each sky built from N-1 frames
        loo_noise_skies = []
        for i in range(len(noise_leveled)):
            others_n = [noise_leveled[j] for j in range(len(noise_leveled)) if j != i]
            var_loo = np.nansum(np.array(others_n)**2, axis=0)
            N_loo = np.sum(np.isfinite(np.array([data_leveled[j] for j in range(len(data_leveled)) if j != i])), axis=0)
            N_loo[N_loo == 0] = 1
            loo_noise_skies.append(
                config['sky_subtraction']['noise_median_factor'] * np.sqrt(var_loo) / N_loo
            )
        
        if verbose:
            logging.info(f"    LOO+IterGauss mode: building leave-one-out skies")
    
    # Formula: (raw_leveled - sky) / flat
    # Note: flat division done AFTER sky subtraction (standard practice)
    skysub_files = []
    
    # Cache loaded flats to avoid reloading
    flat_cache = {}
    
    for i, (data_lev, noise_in, file_info, header) in enumerate(
            zip(data_leveled, noise_leveled, files, headers)):
        
        # Sky subtract: LOO sky or standard sky
        if use_loo_ig:
            data_skysub = data_lev - loo_skies[i]
            noise_skysub = np.sqrt(noise_in**2 + loo_noise_skies[i]**2)
        else:
            data_skysub = data_lev - sky_pattern
            noise_skysub = np.sqrt(noise_in**2 + noise_sky**2)
        
        # Get dither angle for flat field loading
        if 'DITHANGL' in header:
            dither_angle = header['DITHANGL']
        elif 'DWANGLE' in header:
            dither_angle = (header['DWANGLE'] - 72.0) % 360
        else:
            dither_angle = 0.0
        
        dither_angle_rounded = int(round(dither_angle / 72.0) * 72) % 360
        
        # Load flat for this dither position (cached)
        flat = None
        if enable_flat and data_folder and os.path.exists(data_folder):
            cache_key = (filt, dither_angle_rounded)
            if cache_key in flat_cache:
                flat = flat_cache[cache_key]
            else:
                calib = load_calibration_files(data_folder, filt, dither_angle_rounded, config, verbose)
                if calib is not None:
                    flat = calib['flat']
                    flat_cache[cache_key] = flat
        
        # Fallback to unity flat
        if flat is None:
            flat = np.ones_like(data_skysub)
        
        # Apply flat field correction AFTER sky subtraction
        data_final = data_skysub / flat
        noise_final = noise_skysub / flat
        
        # LOO + Iterative Gaussian background correction (after flat division)
        if use_loo_ig:
            ig_n_iter = loo_ig_cfg.get('n_iter', 3)
            ig_base_sigma = loo_ig_cfg.get('base_sigma', 15.0)
            ig_decay = loo_ig_cfg.get('sigma_decay', 0.8)
            ig_sigma_thresh = loo_ig_cfg.get('source_sigma', 1.8)
            ig_grow_base = loo_ig_cfg.get('source_grow_base', 4.0)
            
            current = data_final.copy()
            # Track cumulative variance added by subtracting smooth backgrounds.
            # Each iteration subtracts bg_k = GaussianSmooth(image, σ_k).
            # The variance of bg_k at each pixel ≈ local_var / (2π σ_k²),
            # where the effective number of independent pixels in the kernel
            # is N_eff = 2π σ_k².  We propagate this through the noise map.
            ig_added_var = np.zeros_like(noise_final)
            
            # Build source mask ONCE before the loop.  Re-detecting sources
            # at every iteration caused inconsistent masks across iterations
            # and frames, producing negative "holes" in the coadd.
            work = np.ascontiguousarray(current.astype(np.float32))
            nan_mask = ~np.isfinite(work)
            work[nan_mask] = 0.0
            try:
                bkg = sep.Background(work, bw=64, bh=64, fw=3, fh=3, mask=nan_mask)
                sub = work - bkg.back()
                source_mask = nan_mask.copy()
                objs = sep.extract(sub, ig_sigma_thresh * bkg.globalrms,
                                   minarea=5, mask=nan_mask)
                for obj in objs:
                    try:
                        sep.mask_ellipse(source_mask, obj['x'], obj['y'],
                                         obj['a'], obj['b'], obj['theta'],
                                         r=ig_grow_base)
                    except Exception:
                        pass
            except Exception:
                source_mask = nan_mask.copy()
            
            for it in range(ig_n_iter):
                sig = ig_base_sigma * (ig_decay ** it)
                # Normalized convolution: smooth data and weight separately,
                # then divide.  This interpolates across masked regions
                # using only nearby good pixels.
                bad = source_mask | ~np.isfinite(current)
                filled = current.copy()
                filled[bad] = 0.0
                weight = (~bad).astype(np.float64)
                smooth_data = gaussian_filter(filled.astype(np.float64), sigma=sig)
                smooth_weight = gaussian_filter(weight, sigma=sig)
                bg = np.zeros_like(smooth_data, dtype=np.float32)
                ok = smooth_weight > 0.01
                bg[ok] = (smooth_data[ok] / smooth_weight[ok]).astype(np.float32)
                current = current - bg
                
                # Noise propagation: variance of the smooth bg model
                # var(bg_k) = var(pixel) / N_eff, where N_eff = 2π σ²
                n_eff = 2.0 * np.pi * sig * sig
                ig_added_var += noise_final**2 / n_eff
            
            data_final = current
            # Final noise: original variance + variance from subtracting bg models
            noise_final = np.sqrt(noise_final**2 + ig_added_var)
            
            if verbose and i == 0:  # log once per group
                logging.info(f"    Applied IterGauss: {ig_n_iter} iter, "
                             f"base_sigma={ig_base_sigma}, decay={ig_decay}")
        
        # Handle non-finite values
        noise_final[~np.isfinite(data_final)] = np.nan
        
        skysub_filename = file_info['file'].replace('.fits', '_skysub.fits')
        skysub_path = os.path.join(output_dir, skysub_filename)
        
        skysub_header = header.copy()
        skysub_header['PROCSTAT'] = config['fits_markers']['procstat_reduced']
        skysub_header['PSTATSUB'] = config['fits_markers']['pstatsub_skysub']
        skysub_header['FILENAME'] = skysub_filename
        skysub_header['DATE'] = format_date_like_dateobs(header['DATE-OBS'])
        if use_loo_ig:
            skysub_header['HISTORY'] = f"LOO sky subtracted (median of {len(files)-1} frames)"
            skysub_header['HISTORY'] = f"IterGauss: {loo_ig_cfg.get('n_iter',3)} iter, sigma0={loo_ig_cfg.get('base_sigma',25.0)}"
        else:
            skysub_header['HISTORY'] = f"Sky subtracted (median of {len(files)} frames)"
        skysub_header['HISTORY'] = f"Flat fielded after sky subtraction"
        skysub_header['COMMENT'] = "Processing order: level -> sky -> flat"
        
        # Save data and noise as extensions
        primary_hdu = fits.PrimaryHDU(data_final, header=skysub_header)
        noise_hdu = fits.ImageHDU(noise_final, name='NOISE')
        hdul = fits.HDUList([primary_hdu, noise_hdu])
        hdul.writeto(skysub_path, overwrite=True)
        
        skysub_files.append({
            'file': skysub_filename,
            'path': skysub_path,
            'data': data_final,
            'noise': noise_final,
            'header': skysub_header
        })
    
    return sky_path, skysub_files, group_name


# ============================================================================
# ALIGNMENT AND GEOMETRY UTILITIES
# ============================================================================

def _fit_similarity_transform(src_coords, dst_coords, weights=None,
                              sigma_clip_iters=3, sigma_clip_threshold=3.0):
    """Fit a 2D similarity transform with optional flux weighting and sigma-clipping.

    Finds the best-fit transform mapping src -> dst:
        x_dst = a * x_src - b * y_src + tx
        y_dst = b * x_src + a * y_src + ty

    where a = s*cos(theta), b = s*sin(theta), s = scale, theta = rotation angle.

    Iteratively sigma-clips outlier matches and applies flux-based weights
    for sub-pixel precision.

    Args:
        src_coords: Nx2 array of source (x, y) coordinates
        dst_coords: Nx2 array of destination (x, y) coordinates
        weights: Optional Nx1 array of per-match weights (e.g. sqrt(flux)).
                 If None, all matches weighted equally.
        sigma_clip_iters: Number of sigma-clipping iterations (0 = no clipping)
        sigma_clip_threshold: Reject matches with residual > this many sigma

    Returns:
        dict with keys: a, b, tx, ty, rotation_deg, scale, rms, n_used, inlier_mask
    """
    N = len(src_coords)
    xs, ys = src_coords[:, 0], src_coords[:, 1]
    xd, yd = dst_coords[:, 0], dst_coords[:, 1]

    if weights is None:
        w = np.ones(N)
    else:
        w = np.asarray(weights, dtype=np.float64)
        # Normalise so mean weight = 1 (keeps numerical conditioning stable)
        wmean = np.mean(w)
        if wmean > 0:
            w = w / wmean

    inlier = np.ones(N, dtype=bool)

    for iteration in range(max(1, sigma_clip_iters + 1)):
        idx = np.where(inlier)[0]
        if len(idx) < 4:
            break

        n = len(idx)
        xs_i, ys_i = xs[idx], ys[idx]
        xd_i, yd_i = xd[idx], yd[idx]
        w_i = w[idx]

        # Build weighted design matrix
        A = np.zeros((2 * n, 4))
        A[0::2, 0] = xs_i;   A[0::2, 1] = -ys_i;  A[0::2, 2] = 1.0
        A[1::2, 0] = ys_i;   A[1::2, 1] = xs_i;    A[1::2, 3] = 1.0

        target = np.zeros(2 * n)
        target[0::2] = xd_i
        target[1::2] = yd_i

        # Weight matrix (duplicate each weight for x,y equations)
        W_diag = np.empty(2 * n)
        W_diag[0::2] = w_i
        W_diag[1::2] = w_i

        # Weighted least squares: (A^T W A) p = A^T W b
        AW = A * W_diag[:, None]
        params, _, _, _ = np.linalg.lstsq(AW, target * W_diag, rcond=None)

        # Compute per-match residuals on ALL pairs (for sigma-clipping)
        A_full = np.zeros((2 * N, 4))
        A_full[0::2, 0] = xs;   A_full[0::2, 1] = -ys;  A_full[0::2, 2] = 1.0
        A_full[1::2, 0] = ys;   A_full[1::2, 1] = xs;    A_full[1::2, 3] = 1.0
        target_full = np.zeros(2 * N)
        target_full[0::2] = xd
        target_full[1::2] = yd

        predicted = A_full @ params
        res = target_full - predicted
        # Per-match scalar residual = sqrt(res_x^2 + res_y^2)
        res_per_match = np.sqrt(res[0::2]**2 + res[1::2]**2)

        if iteration < sigma_clip_iters:
            med = np.median(res_per_match[inlier])
            mad = np.median(np.abs(res_per_match[inlier] - med))
            sigma = mad * 1.4826  # MAD -> sigma
            if sigma < 1e-6:
                sigma = 1e-6
            inlier = res_per_match < med + sigma_clip_threshold * sigma

    a, b, tx, ty = params
    scale = np.sqrt(a**2 + b**2)
    rotation_deg = np.degrees(np.arctan2(b, a))

    # RMS on inliers
    n_used = int(np.sum(inlier))
    if n_used > 0:
        rms = np.sqrt(np.mean(res_per_match[inlier]**2))
    else:
        rms = np.sqrt(np.mean(res_per_match**2))
        n_used = N

    return {'a': a, 'b': b, 'tx': tx, 'ty': ty,
            'rotation_deg': rotation_deg, 'scale': scale,
            'rms': rms, 'n_used': n_used, 'inlier_mask': inlier}


def refine_alignment_with_crossmatch(skysub_files, config, system, verbose=False):
    """Refine alignment using source detection, iterative cross-matching,
    and sigma-clipped shift fitting.
    
    Algorithm per image:
    1. Compute blind shifts from dither geometry (initial guess)
    2. Detect sources in all images (capped to brightest N)
    3. Iteration 1: apply blind shift, cross-match with KDTree,
       fit sigma-clipped median shift (or optionally similarity transform)
    4. Iteration 2: re-project through fitted shift, cross-match with
       tighter tolerance, re-fit.  Recovers borderline matches.
    5. Return refined transforms if enough matches, blind shift otherwise.
    
    By default uses shift-only fitting (2 parameters: dx, dy) which is 2x
    more precise than similarity transform (4 params: a, b, tx, ty) for
    REMIR's wedge-prism dither pattern where inter-frame rotation is <0.02°.
    Similarity transform fitting available via fit_mode: similarity config.
    
    Args:
        skysub_files: List of dicts with 'data', 'noise', 'header', 'path'
        config: Pipeline configuration dictionary
        system: 'old' or 'new' for dither geometry parameters
        verbose: Enable verbose logging
    
    Returns:
        list or None: List of transform dicts for each image if successful, None if failed.
                     Each dict has keys: a, b, tx, ty (affine transform parameters),
                     rotation_deg, scale.  The transform maps image → template:
                         x_out = a * x - b * y + tx
                         y_out = b * x + a * y + ty
                     First element is always the identity for the template.
    """
    refine_cfg = config['alignment'].get('refinement', {})
    
    # Check if refinement is enabled
    if not refine_cfg.get('enabled', False):
        return None
    
    if verbose:
        logging.info("  Attempting alignment refinement with cross-matching...")
    
    # Extract parameters
    min_pixels = refine_cfg.get('min_pixels', 5)
    threshold_sigma = refine_cfg.get('threshold_sigma', 2.0)
    pix_tol = refine_cfg.get('pix_tol', 2.0)
    min_matches = refine_cfg.get('min_matches', 4)
    min_match_fraction = refine_cfg.get('min_match_fraction', 0.15)
    accept_rms_px = refine_cfg.get('accept_rms_px', 1.5)
    max_sources = refine_cfg.get('max_sources', 50)
    n_refine_iters = refine_cfg.get('n_refine_iters', 2)
    sigma_clip_iters = refine_cfg.get('sigma_clip_iters', 3)
    sigma_clip_threshold = refine_cfg.get('sigma_clip_threshold', 3.0)
    fit_mode = refine_cfg.get('fit_mode', 'shift')  # 'shift' or 'similarity'
    
    if verbose:
        logging.info(f"  Alignment fit mode: {fit_mode}")
    
    # First compute blind shifts from dither geometry
    alignment_config = config['alignment'][system]
    theta_n = alignment_config['theta_n']
    r_n = alignment_config['r_n']
    theta_offset = alignment_config['theta_offset']
    dithangl_key = alignment_config['dithangl_key']
    base_angle = config['alignment'].get('base_angle', 72)
    
    theta_degrees_base = []
    for file_info in skysub_files:
        header = file_info['header']
        dith_angle = header.get(dithangl_key, 0)
        theta = dith_angle + base_angle + theta_offset
        theta_degrees_base.append(theta)
    
    theta_degrees_base = np.array(theta_degrees_base)
    theta_radians_n = np.deg2rad(90 - theta_degrees_base - theta_n)
    x_n = r_n * np.cos(theta_radians_n)
    y_n = r_n * np.sin(theta_radians_n)
    
    template_idx = 0
    x_n_tem = x_n - x_n[template_idx]
    y_n_tem = y_n - y_n[template_idx]
    blind_shifts = [(-x_n_tem[i], -y_n_tem[i]) for i in range(len(skysub_files))]
    
    # Detect sources in all images (capped to brightest N)
    all_sources = []
    for idx, file_info in enumerate(skysub_files):
        try:
            sources, _, _ = detect_sources(
                file_info['data'], config, 
                npix_min=min_pixels, 
                threshold_sigma=threshold_sigma
            )
            # Cap to brightest N (sources already sorted by mag, brightest first)
            if len(sources) > max_sources:
                sources = sources[:max_sources]
            all_sources.append(sources)
            if verbose:
                logging.info(f"    Image {idx}: Detected {len(sources)} sources"
                           + (f" (capped to {max_sources})" if len(sources) == max_sources else ""))
        except Exception as e:
            if verbose:
                logging.info(f"    Image {idx}: Source detection failed: {e}")
            return None
    
    # Template sources
    template_sources = all_sources[template_idx]
    if len(template_sources) < min_matches:
        if verbose:
            logging.info(f"    Template has only {len(template_sources)} sources (need >= {min_matches})")
        return None
    
    template_coords = np.array([[s['x'], s['y']] for s in template_sources])
    template_fluxes = np.array([max(s.get('flux', 1.0), 1.0) for s in template_sources])
    
    # Refine shifts for each image
    refined_shifts = []
    
    for idx, file_info in enumerate(skysub_files):
        if idx == template_idx:
            refined_shifts.append({'a': 1.0, 'b': 0.0, 'tx': 0.0, 'ty': 0.0, 'rotation_deg': 0.0, 'scale': 1.0})
            continue
        
        img_sources = all_sources[idx]
        if len(img_sources) < min_matches:
            if verbose:
                logging.info(f"    Image {idx}: Only {len(img_sources)} sources, using blind shift")
            bx, by = blind_shifts[idx]
            refined_shifts.append({'a': 1.0, 'b': 0.0, 'tx': bx, 'ty': by, 'rotation_deg': 0.0, 'scale': 1.0})
            continue
        
        img_coords = np.array([[s['x'], s['y']] for s in img_sources])
        img_fluxes = np.array([max(s.get('flux', 1.0), 1.0) for s in img_sources])
        
        # Iterative refinement: start with blind shift, then re-match through
        # the fitted transform with tighter tolerance.
        current_transform = None
        blind_dx, blind_dy = blind_shifts[idx]
        best_transform = None
        
        for refine_iter in range(n_refine_iters):
            # Tolerance: wider on first pass, tighter on subsequent
            tol = pix_tol if refine_iter == 0 else pix_tol * 0.6
            
            if current_transform is None:
                # First iteration: use blind shift
                projected = img_coords + np.array([blind_dx, blind_dy])
            else:
                # Subsequent iterations: project through current transform
                a_c, b_c = current_transform['a'], current_transform['b']
                tx_c, ty_c = current_transform['tx'], current_transform['ty']
                px = a_c * img_coords[:, 0] - b_c * img_coords[:, 1] + tx_c
                py = b_c * img_coords[:, 0] + a_c * img_coords[:, 1] + ty_c
                projected = np.column_stack([px, py])
            
            # Cross-match with template using KDTree
            tree = cKDTree(template_coords)
            dists, indices = tree.query(projected, distance_upper_bound=tol)
            
            matched_mask = dists < tol
            n_matched = np.sum(matched_mask)
            match_fraction = n_matched / len(template_sources) if len(template_sources) > 0 else 0
            
            if n_matched < min_matches or match_fraction < min_match_fraction:
                if refine_iter == 0:
                    # First pass failed — rescue with progressively wider tolerance
                    rescued = False
                    for rescue_factor in [2, 4, 8]:
                        rescue_tol = pix_tol * rescue_factor
                        dists_r, indices_r = tree.query(projected, distance_upper_bound=rescue_tol)
                        mask_r = dists_r < rescue_tol
                        n_r = np.sum(mask_r)
                        frac_r = n_r / len(template_sources) if len(template_sources) > 0 else 0
                        if n_r >= min_matches and frac_r >= min_match_fraction:
                            matched_mask = mask_r
                            indices = indices_r
                            n_matched = n_r
                            match_fraction = frac_r
                            rescued = True
                            if verbose:
                                logging.info(f"    Image {idx}: Rescued with {rescue_factor}x tolerance "
                                           f"({rescue_tol:.1f}px): {n_r} matches ({frac_r:.1%})")
                            break
                    if not rescued:
                        break
                else:
                    # Keep previous iteration's result
                    break
            
            # Get matched pairs
            matched_img_orig = img_coords[matched_mask]
            matched_tmpl = template_coords[indices[matched_mask]]
            
            if fit_mode == 'similarity':
                # Full similarity transform (rotation + scale + translation)
                w_img = np.sqrt(img_fluxes[matched_mask])
                w_tmpl = np.sqrt(template_fluxes[indices[matched_mask]])
                match_weights = np.sqrt(w_img * w_tmpl)
                transform = _fit_similarity_transform(
                    matched_img_orig, matched_tmpl,
                    weights=match_weights,
                    sigma_clip_iters=sigma_clip_iters,
                    sigma_clip_threshold=sigma_clip_threshold
                )
            else:
                # Shift-only fitting: sigma-clipped median of (template - shifted) offsets
                if current_transform is None:
                    shifted_coords = matched_img_orig + np.array([blind_dx, blind_dy])
                else:
                    shifted_coords = np.column_stack([
                        current_transform['a'] * matched_img_orig[:, 0] - current_transform['b'] * matched_img_orig[:, 1] + current_transform['tx'],
                        current_transform['b'] * matched_img_orig[:, 0] + current_transform['a'] * matched_img_orig[:, 1] + current_transform['ty']
                    ])
                diff = matched_tmpl - shifted_coords
                inlier = np.ones(len(diff), dtype=bool)
                for _sc in range(sigma_clip_iters):
                    if np.sum(inlier) < 3:
                        break
                    med = np.median(diff[inlier], axis=0)
                    d = np.sqrt(np.sum((diff[inlier] - med)**2, axis=1))
                    mad = np.median(d) * 1.4826
                    if mad < 1e-6:
                        break
                    inlier_sub = d < sigma_clip_threshold * mad
                    full_inlier = np.where(inlier)[0][inlier_sub]
                    inlier = np.zeros(len(diff), dtype=bool)
                    inlier[full_inlier] = True
                ddx, ddy = np.median(diff[inlier], axis=0)
                if current_transform is None:
                    total_tx = blind_dx + ddx
                    total_ty = blind_dy + ddy
                else:
                    total_tx = current_transform['tx'] + ddx
                    total_ty = current_transform['ty'] + ddy
                # Compute RMS on residuals after subtracting fitted shift
                residuals = diff[inlier] - np.array([ddx, ddy])
                res = np.sqrt(residuals[:, 0]**2 + residuals[:, 1]**2)
                rms_val = np.sqrt(np.mean(res**2)) if len(res) > 0 else 999.0
                transform = {
                    'a': 1.0, 'b': 0.0, 'tx': total_tx, 'ty': total_ty,
                    'rotation_deg': 0.0, 'scale': 1.0,
                    'rms': rms_val, 'n_used': int(np.sum(inlier))
                }
            
            rms = transform['rms']
            n_used = transform['n_used']
            
            if rms <= accept_rms_px:
                current_transform = transform
                best_transform = transform
                best_n_matched = n_matched
                best_n_used = n_used
        
        # Use best transform, or fall back to blind shift
        if best_transform is None:
            if verbose:
                n_matched_log = np.sum(dists < pix_tol) if 'dists' in dir() else 0
                match_frac_log = n_matched_log / len(template_sources) if len(template_sources) > 0 else 0
                logging.info(f"    Image {idx}: Only {n_matched_log} matches ({match_frac_log:.1%}), using blind shift")
            bx, by = blind_shifts[idx]
            refined_shifts.append({'a': 1.0, 'b': 0.0, 'tx': bx, 'ty': by, 'rotation_deg': 0.0, 'scale': 1.0})
            continue
        
        a, b = best_transform['a'], best_transform['b']
        rot_deg = best_transform['rotation_deg']
        scale = best_transform['scale']
        rms = best_transform['rms']
        
        # Effective translation at image center (for logging)
        ny, nx = file_info['data'].shape
        cx, cy = nx / 2.0, ny / 2.0
        eff_dx = a * cx - b * cy + best_transform['tx'] - cx
        eff_dy = b * cx + a * cy + best_transform['ty'] - cy
        
        refined_shifts.append({
            'a': a, 'b': b, 'tx': best_transform['tx'], 'ty': best_transform['ty'],
            'rotation_deg': rot_deg, 'scale': scale
        })
        
        if verbose:
            if fit_mode == 'similarity':
                logging.info(f"    Image {idx}: Refined shift dx={eff_dx:.3f}, dy={eff_dy:.3f} "
                            f"(rot={rot_deg:+.4f} deg, scale={scale:.5f}, "
                            f"matches={best_n_matched}, used={best_n_used}, RMS={rms:.3f}px)")
            else:
                logging.info(f"    Image {idx}: Refined shift dx={eff_dx:.3f}, dy={eff_dy:.3f} "
                            f"(matches={best_n_matched}, used={best_n_used}, RMS={rms:.3f}px)")
    
    if verbose:
        logging.info("  Alignment refinement complete!")
    
    return refined_shifts


def drizzle_image(data, noise, transform, output_shape=None, pixfrac=1.0):
    """Apply drizzling with full affine transform (rotation + scale + translation).
    
    VECTORIZED implementation. Each input pixel at position (x, y) maps to output:
        x_out = a * x - b * y + tx
        y_out = b * x + a * y + ty
    
    For pure translation: a=1, b=0, tx=dx, ty=dy.
    
    Unlike a two-step approach (interpolation pre-correction + shift drizzle),
    this applies the full geometric transform in a single drizzle pass — no
    double interpolation, preserving image sharpness.  This is the same
    principle as AQuA/PREPROCESS's polygon-intersection remapping: each input
    pixel's flux is distributed to output pixels based on geometric overlap.
    
    The pixel drop is an axis-aligned square of side ``pixfrac`` centred at the
    transformed position.  For rotations below ~5 degrees the error from
    ignoring the pixel-shape rotation is < 0.05 pixels per edge — negligible
    compared to typical REMIR seeing.
    
    Args:
        data: Input 2D array to be drizzled
        noise: Input 2D noise array
        transform: dict with affine transform parameters:
            - 'a', 'b', 'tx', 'ty' for full similarity transform, or
            - 'dx', 'dy' for pure translation (a=1, b=0)
        output_shape: Shape of output array, defaults to input shape
        pixfrac: Pixel fraction (1.0 = full pixel, <1.0 = shrink for sharper PSF)
    
    Returns:
        tuple: (drizzled_data, drizzled_noise, weight_map)
    """
    if output_shape is None:
        output_shape = data.shape
    
    ny_out, nx_out = output_shape
    ny_in, nx_in = data.shape
    
    # Extract affine transform parameters
    a = transform.get('a', 1.0)
    b = transform.get('b', 0.0)
    tx = transform.get('tx', transform.get('dx', 0.0))
    ty = transform.get('ty', transform.get('dy', 0.0))
    
    # Initialize output arrays
    flux_out = np.zeros(output_shape, dtype=np.float64)
    var_out = np.zeros(output_shape, dtype=np.float64)
    weight_out = np.zeros(output_shape, dtype=np.float64)
    
    # Create coordinate grids for all input pixels
    iy_in, ix_in = np.mgrid[0:ny_in, 0:nx_in]
    ix = ix_in.astype(np.float64)
    iy = iy_in.astype(np.float64)
    
    # Transform input pixel centers to output coordinates (full affine)
    x_center = a * ix - b * iy + tx
    y_center = b * ix + a * iy + ty
    
    # Pixel drop boundaries (axis-aligned square of side pixfrac)
    half_pix = pixfrac / 2.0
    x_min = x_center - half_pix
    x_max = x_center + half_pix
    y_min = y_center - half_pix
    y_max = y_center + half_pix
    
    # Get valid (non-NaN) pixels
    valid = np.isfinite(data)
    flux = np.where(valid, data, 0).astype(np.float64)
    var = np.where(valid, noise**2, 0).astype(np.float64)
    
    # For pixfrac <= 1, each input pixel overlaps at most 4 output pixels (2x2).
    # Output pixel i extends from [i-0.5, i+0.5], so position p belongs to
    # pixel floor(p + 0.5).  This is the correct mapping for pixel centers at
    # integer coordinates, fixing a boundary rounding issue.
    ox_lo = np.floor(x_min + 0.5).astype(np.int32)
    ox_hi = np.floor(x_max + 0.5).astype(np.int32)
    oy_lo = np.floor(y_min + 0.5).astype(np.int32)
    oy_hi = np.floor(y_max + 0.5).astype(np.int32)
    
    # Process the 4 possible output pixels per input pixel (2x2 grid)
    for oy_idx, ox_idx in [(oy_lo, ox_lo), (oy_lo, ox_hi), (oy_hi, ox_lo), (oy_hi, ox_hi)]:
        # Check bounds
        in_bounds = (ox_idx >= 0) & (ox_idx < nx_out) & (oy_idx >= 0) & (oy_idx < ny_out) & valid
        
        if not np.any(in_bounds):
            continue
        
        # Output pixel boundaries (centered at integer coordinates)
        ox_min_grid = ox_idx - 0.5
        ox_max_grid = ox_idx + 0.5
        oy_min_grid = oy_idx - 0.5
        oy_max_grid = oy_idx + 0.5
        
        # Compute overlap rectangle
        overlap_x_min = np.maximum(x_min, ox_min_grid)
        overlap_x_max = np.minimum(x_max, ox_max_grid)
        overlap_y_min = np.maximum(y_min, oy_min_grid)
        overlap_y_max = np.minimum(y_max, oy_max_grid)
        
        # Overlap area (zero if no overlap)
        overlap_x = np.maximum(0, overlap_x_max - overlap_x_min)
        overlap_y = np.maximum(0, overlap_y_max - overlap_y_min)
        overlap_area = overlap_x * overlap_y
        
        # Mask: in bounds, valid, and has overlap
        contrib_mask = in_bounds & (overlap_area > 0)
        
        if not np.any(contrib_mask):
            continue
        
        # Extract contributing pixels
        oy_flat = oy_idx[contrib_mask]
        ox_flat = ox_idx[contrib_mask]
        area_flat = overlap_area[contrib_mask]
        flux_flat = flux[contrib_mask]
        var_flat = var[contrib_mask]
        
        # Accumulate using np.add.at (handles duplicate indices correctly)
        np.add.at(flux_out, (oy_flat, ox_flat), flux_flat * area_flat)
        np.add.at(var_out, (oy_flat, ox_flat), var_flat * area_flat**2)
        np.add.at(weight_out, (oy_flat, ox_flat), area_flat)
    
    # Normalize by weight (avoid division by zero)
    mask = weight_out > 1e-10
    drizzled_data = np.full(output_shape, np.nan, dtype=np.float32)
    drizzled_noise = np.full(output_shape, np.nan, dtype=np.float32)
    
    drizzled_data[mask] = (flux_out[mask] / weight_out[mask]).astype(np.float32)
    drizzled_noise[mask] = (np.sqrt(var_out[mask]) / weight_out[mask]).astype(np.float32)
    
    return drizzled_data, drizzled_noise, weight_out.astype(np.float32)


def align_frames(skysub_files, config, system, output_dir, verbose=False):
    """Align sky-subtracted frames using dither geometry (optionally refined).
    
    REMIR uses a rotating wedge prism that introduces predictable dither
    offsets based on the dither angle. This function calculates the
    expected pixel shifts and applies a single-pass affine drizzle to
    align all frames to the first frame (template).
    
    Optional refinement: If enabled in config, attempts to refine the alignment
    by detecting sources and cross-matching between images, fitting a full
    similarity transform (rotation + scale + translation). Falls back to blind
    dither geometry if refinement fails.
    
    Shift calculation (blind mode):
        dx = r_n * cos(dith_angle + theta_offset + theta_n + base_angle)
        dy = r_n * sin(dith_angle + theta_offset + theta_n + base_angle)
        shift = (dx - dx_template, dy - dy_template)
    
    The affine drizzle distributes each input pixel's flux among output pixels
    based on geometric overlap, handling rotation + scale + translation in one
    pass with no double interpolation. This follows the same principle as
    AQuA/PREPROCESS's polygon-intersection remapping.
    
    Args:
        skysub_files: List of dicts with 'data', 'noise', 'header', 'path'
        config: Pipeline configuration dictionary
        system: 'old' or 'new' for system-specific alignment parameters
        output_dir: Directory to write aligned files
        verbose: Enable verbose logging
    
    Returns:
        list: Dicts with aligned file info including paths and offsets
    """
    alignment_config = config['alignment'][system]
    
    theta_n = alignment_config['theta_n']
    r_n = alignment_config['r_n']
    theta_offset = alignment_config['theta_offset']
    dithangl_key = alignment_config['dithangl_key']
    base_angle = config['alignment'].get('base_angle', 72)
    
    # Try refinement if enabled
    refined_shifts = None
    alignment_method = "blind dither geometry"
    
    refine_cfg = config['alignment'].get('refinement', {})
    if refine_cfg.get('enabled', False):
        refined_shifts = refine_alignment_with_crossmatch(skysub_files, config, system, verbose=verbose)
        if refined_shifts is not None:
            alignment_method = "similarity-transform refinement"
        else:
            if verbose:
                logging.info("  Refinement failed, falling back to blind dither geometry")
            if not refine_cfg.get('fallback_to_blind', True):
                raise RuntimeError("Alignment refinement failed and fallback is disabled")
    
    # If no refinement, compute blind shifts from dither geometry
    if refined_shifts is None:
        # Extract dither angles from headers
        theta_degrees_base = []
        for file_info in skysub_files:
            header = file_info['header']
            dith_angle = header.get(dithangl_key, 0)
            theta = dith_angle + base_angle + theta_offset
            theta_degrees_base.append(theta)
        
        # Calculate absolute dithering positions
        theta_degrees_base = np.array(theta_degrees_base)
        theta_radians_n = np.deg2rad(90 - theta_degrees_base - theta_n)
        x_n = r_n * np.cos(theta_radians_n)
        y_n = r_n * np.sin(theta_radians_n)
        
        # Use first file as alignment template
        template_idx = 0
        x_n_tem = x_n - x_n[template_idx]
        y_n_tem = y_n - y_n[template_idx]
        
        # Compute shifts (note: negative because we shift image, not coordinate system)
        refined_shifts = [{'a': 1.0, 'b': 0.0, 'tx': -x_n_tem[i], 'ty': -y_n_tem[i]}
                          for i in range(len(skysub_files))]
    
    # Apply transforms to all images using drizzling (flux-preserving, no interpolation smoothing)
    template_idx = 0
    aligned_files = []
    
    pixfrac = config['alignment'].get('drizzle_pixfrac', 1.0)  # Pixel fraction for drizzling
    
    if verbose:
        logging.info(f"  Applying affine drizzle (pixfrac={pixfrac})...")
    
    for idx, file_info in enumerate(skysub_files):
        header = file_info['header']
        data = file_info['data']
        noise = file_info['noise']
        
        # Get full affine transform for this image
        shift_info = refined_shifts[idx]
        
        # Apply drizzling with full affine transform in a single pass.
        # Rotation + scale + translation are handled together — no separate
        # interpolation step, no double-smoothing.
        data_aligned, noise_aligned, weight = drizzle_image(
            data, noise, shift_info, output_shape=data.shape, pixfrac=pixfrac
        )
        
        # Compute effective shift at image center (for logging / FITS headers)
        a_t = shift_info.get('a', 1.0)
        b_t = shift_info.get('b', 0.0)
        tx_t = shift_info.get('tx', shift_info.get('dx', 0.0))
        ty_t = shift_info.get('ty', shift_info.get('dy', 0.0))
        ny, nx = data.shape
        cx, cy = nx / 2.0, ny / 2.0
        dx = (a_t - 1) * cx - b_t * cy + tx_t
        dy = b_t * cx + (a_t - 1) * cy + ty_t
        rot_val = shift_info.get('rotation_deg', 0.0)
        scale_val = shift_info.get('scale', 1.0)
        
        aligned_filename = file_info['file'].replace('_skysub.fits', '_skysub_aligned.fits')
        aligned_path = os.path.join(output_dir, aligned_filename)
        
        aligned_header = header.copy()
        aligned_header['PROCSTAT'] = config['fits_markers']['procstat_reduced']
        aligned_header['PSTATSUB'] = config['fits_markers']['pstatsub_aligned']
        aligned_header['FILENAME'] = aligned_filename
        aligned_header['DATE'] = format_date_like_dateobs(header['DATE-OBS'])
        aligned_header['HISTORY'] = f"Aligned to template {skysub_files[template_idx]['file']}"
        aligned_header['HISTORY'] = f"Applied shift: dx={dx:.3f}, dy={dy:.3f} pixels"
        aligned_header['HISTORY'] = f"Alignment method: {alignment_method} (affine drizzle)"
        aligned_header['HISTORY'] = f"Drizzle pixfrac: {pixfrac}"
        if abs(rot_val) > 1e-6 or abs(scale_val - 1.0) > 1e-6:
            aligned_header['HISTORY'] = f"Affine drizzle: rot={rot_val:+.4f} deg, scale={scale_val:.5f}"
        if alignment_method == "blind dither geometry":
            aligned_header['HISTORY'] = f"Dithering pattern: theta_n={theta_n}, r_n={r_n}, theta_off={theta_offset}"
        aligned_header['ALIGNED'] = (True, 'File has been aligned')
        aligned_header['ALIGNMTH'] = (f"{alignment_method} (affine drizzle)", 'Alignment method used')
        aligned_header['DRZLPIXF'] = (pixfrac, 'Drizzle pixel fraction')
        aligned_header['TEMPLATE'] = (idx == template_idx, 'This is the template file')
        if idx != template_idx:
            aligned_header['SHIFT_X'] = (dx, 'Effective X shift at image center [pixels]')
            aligned_header['SHIFT_Y'] = (dy, 'Effective Y shift at image center [pixels]')
        if abs(rot_val) > 1e-6:
            aligned_header['ROT_DEG'] = (rot_val, 'Rotation angle [degrees]')
        if abs(scale_val - 1.0) > 1e-6:
            aligned_header['SCALE'] = (scale_val, 'Scale factor')
        
        primary_hdu = fits.PrimaryHDU(data_aligned, header=aligned_header)
        noise_hdu = fits.ImageHDU(noise_aligned, name='NOISE')
        weight_hdu = fits.ImageHDU(weight, name='WEIGHT')
        hdul = fits.HDUList([primary_hdu, noise_hdu, weight_hdu])
        hdul.writeto(aligned_path, overwrite=True)
        
        aligned_files.append({
            'file': aligned_filename,
            'path': aligned_path,
            'data': data_aligned,
            'noise': noise_aligned,
            'weight': weight,
            'header': aligned_header
        })
    
    return aligned_files

def coadd_aligned_frames(aligned_files, group_name, output_dir, is_incomplete, config, verbose=False):
    """Combine aligned frames into a single coadded image.
    
    Performs inverse-variance weighted combination of aligned frames
    with iterative positive-outlier sigma-clipping (cosmic ray rejection).
    Drizzle-edge pixels with higher noise automatically receive lower weight,
    producing cleaner coadds with optimal signal-to-noise.
    
    Algorithm:
    1. Compute inverse-variance weights: w_i = 1 / noise_i^2
    2. If >= 3 frames: iteratively clip pixels brighter than sigma_clip × σ
       above the weighted mean (positive outliers only)
    3. Weighted mean: coadd = sum(data_i * w_i) / sum(w_i)
    4. Optimal noise: noise = sqrt(1 / sum(w_i))
    
    The coadd header includes:
    - DITHID=99: Marker for coadded images
    - EXPTIME: Total exposure time (sum of all frames)
    - NCOADD: Number of coadded frames
    - DATE-OBS: Average of observation timestamps
    - INCOMP: Flag indicating if dither sequence was incomplete
    
    Args:
        aligned_files: List of dicts with 'data', 'noise', 'weight', 'header', 'path'
        group_name: String identifier for the group (used in filename)
        output_dir: Directory to write coadd FITS file
        is_incomplete: True if this is from an incomplete dither sequence
        config: Pipeline configuration dictionary
        verbose: Enable verbose logging
    
    Returns:
        str: Path to the written coadd FITS file
    """
    data_stack = np.array([f['data'] for f in aligned_files])
    noise_stack = np.array([f['noise'] for f in aligned_files])
    weight_stack = np.array([f.get('weight', np.ones_like(f['data'])) for f in aligned_files])
    
    n_frames = len(aligned_files)
    
    # ========== INVERSE-VARIANCE WEIGHTED CO-ADDITION ==========
    # Drizzle-edge pixels have higher noise -> automatically get lower weight
    # This produces a cleaner coadd with proper noise properties
    var_stack = noise_stack**2
    valid = np.isfinite(data_stack) & np.isfinite(noise_stack) & (var_stack > 0)
    inv_var = np.where(valid, 1.0 / var_stack, 0.0)
    
    # Sigma-clipping: reject positive outliers (cosmic rays) when >= 3 frames
    coadd_cfg = config.get('coadd', {})
    sigma_clip_thresh = coadd_cfg.get('sigma_clip', 4.0)
    sigma_clip_iters = coadd_cfg.get('sigma_clip_iters', 2)
    
    if n_frames >= 3 and sigma_clip_iters > 0:
        for _ in range(sigma_clip_iters):
            sum_inv_var = np.sum(inv_var, axis=0)
            safe = sum_inv_var > 1e-10
            weighted_mean = np.where(safe,
                                     np.sum(np.where(valid, data_stack * inv_var, 0), axis=0) /
                                     np.clip(sum_inv_var, 1e-10, None), 0)
            residual = data_stack - weighted_mean[np.newaxis, :, :]
            # Per-pixel weighted variance
            n_valid = np.sum(valid, axis=0)
            weighted_var = np.where((safe) & (n_valid > 1),
                                    np.sum(np.where(valid, inv_var * residual**2, 0), axis=0) /
                                    np.clip(sum_inv_var, 1e-10, None), 0)
            weighted_std = np.sqrt(weighted_var)
            # Symmetric clipping: reject both positive (cosmic rays) and negative
            # (sky-subtraction holes) outliers beyond sigma_clip_thresh × σ
            outlier = valid & (np.abs(residual) > sigma_clip_thresh * weighted_std[np.newaxis, :, :])
            new_valid = valid & ~outlier
            new_inv_var = np.where(new_valid, 1.0 / var_stack, 0.0)
            if np.array_equal(new_valid, valid):
                break
            valid = new_valid
            inv_var = new_inv_var
    
    sum_inv_var = np.sum(inv_var, axis=0)
    good = sum_inv_var > 1e-10
    
    data_coadd = np.full(data_stack.shape[1:], np.nan, dtype=np.float32)
    noise_coadd = np.full_like(data_coadd, np.nan)
    
    # Weighted mean: sum(data_i / sigma_i^2) / sum(1 / sigma_i^2)
    data_coadd[good] = (np.sum(np.where(valid, data_stack * inv_var, 0), axis=0)[good] /
                         sum_inv_var[good]).astype(np.float32)
    
    # Optimal noise: sqrt(1 / sum(1 / sigma_i^2))
    noise_coadd[good] = np.sqrt(1.0 / sum_inv_var[good]).astype(np.float32)
    
    # Coverage weight map (sum of drizzle weights across all frames)
    weight_coadd = np.sum(np.where(valid, weight_stack, 0), axis=0).astype(np.float32)
    
    coadd_filename = f"{group_name}.fits"
    coadd_path = os.path.join(output_dir, coadd_filename)
    
    headers = [f['header'] for f in aligned_files]
    coadd_header = headers[0].copy()
    
    coadd_header['DITHID'] = config['fits_markers']['dithid_coadd']
    coadd_header['PROCSTAT'] = config['fits_markers']['procstat_reduced']
    coadd_header['PSTATSUB'] = config['fits_markers']['pstatsub_coadd']
    coadd_header['FILENAME'] = coadd_filename
    coadd_header['DATE-OBS'] = average_date_obs([h['DATE-OBS'] for h in headers])
    
    exptimes = [h.get('EXPTIME', 0) for h in headers]
    coadd_header['EXPTIME'] = (float(np.sum(exptimes)), 'Total exposure time (sum of all frames) [s]')
    coadd_header['NCOADD'] = (n_frames, 'Number of coadded frames')
    
    coadd_header['DATE'] = format_date_like_dateobs(coadd_header['DATE-OBS'])
    coadd_header['INCOMP'] = 1 if is_incomplete else 0
    coadd_header['HISTORY'] = f"Coadded from {n_frames} aligned frames (inverse-variance weighted mean)"
    coadd_header['COMMENT'] = "Inverse-variance weighted mean; pixel values at single-frame level"
    coadd_header['COMMENT'] = "WEIGHT extension = sum of drizzle coverage weights per pixel"
    coadd_header['COMMENT'] = "Note: GAIN and RON values are from individual frames, not effective coadd values"
    
    primary_hdu = fits.PrimaryHDU(data_coadd, header=coadd_header)
    noise_hdu = fits.ImageHDU(noise_coadd, name='NOISE')
    weight_hdu = fits.ImageHDU(weight_coadd, name='WEIGHT')
    hdul = fits.HDUList([primary_hdu, noise_hdu, weight_hdu])
    hdul.writeto(coadd_path, overwrite=True)
    # Count coadds created
    STATS['coadds'] += 1
    
    return coadd_path

# ============================================================================
# STEP 4: CATALOG DOWNLOAD
# ============================================================================

def download_catalog(ra, dec, radius_arcmin, catalog_type='2mass', config=None):
    """Download stellar catalog from INAF web service.
    
    Queries the INAF catalog service (cats.oas.inaf.it) for 2MASS or VSX
    catalogs within a cone search region. Implements exponential backoff
    retry logic for network resilience.
    
    Args:
        ra: Right Ascension in degrees (J2000)
        dec: Declination in degrees (J2000)
        radius_arcmin: Search radius in arcminutes
        catalog_type: '2mass' for 2MASS point source catalog,
                      'vsx' for AAVSO Variable Star Index
        config: Pipeline configuration for timeout/retry settings
    
    Returns:
        pandas.DataFrame: Catalog data with columns depending on type:
            - 2MASS: RAJ2000, DEJ2000, Jmag, Hmag, Kmag, e_Jmag, e_Hmag, e_Kmag
            - VSX: Name, RAJ2000, DEJ2000, Type, Period, etc.
    
    Raises:
        Exception: If all retry attempts fail
    """
    radius_deg = radius_arcmin / 60.0
    
    # Get download parameters from config
    cat_cfg = config['astrometry']['catalog'] if config else {}
    timeout = cat_cfg.get('download_timeout', 30)
    limit = cat_cfg.get('download_limit', 10000)
    max_retries = cat_cfg.get('download_retries', 3)
    
    if catalog_type == '2mass':
        url = (
            f"https://cats.oas.inaf.it/2mass/"
            f"radius={radius_deg}&ra={ra}&dec={dec}&limit={limit}&csv"
        )
    elif catalog_type == 'vsx':
        url = (
            f"https://cats.oas.inaf.it/vsx/"
            f"radius={radius_deg}&ra={ra}&dec={dec}&limit={limit}&csv"
        )
    else:
        raise ValueError(f"Unknown catalog type: {catalog_type}")
    
    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            
            # Parse CSV
            df = pd.read_csv(StringIO(response.text))
            
            return df
        
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s...
                logging.warning(f"  Retry {attempt + 1}/{max_retries} for {catalog_type} catalog (waiting {wait_time}s): {e}")
                time.sleep(wait_time)
    
    logging.error(f"Failed to download {catalog_type} catalog after {max_retries} attempts: {last_error}")
    return None

def process_catalog(cat_2mass, cat_vsx, config):
    """Process and merge catalogs."""
    # Add variable column
    default_mag = config['astrometry']['catalog']['default_mag']
    default_err = config['astrometry']['catalog']['default_error']
    vsx_tol = config['astrometry']['catalog']['vsx_match_arcsec']
    
    cat_2mass['variable'] = 0
    
    # Preserve star/galaxy separation columns from 2MASS
    # ext_key: extended source association key (non-empty = likely galaxy/extended)
    # gal_contam: galaxy contamination flag (0 = no contamination)
    # ph_qual: photometric quality flag per band (A=best, E=worst; 'AAA' = good JHK)
    # prox: distance to nearest 2MASS neighbor [arcsec] (large = isolated)
    if 'ext_key' not in cat_2mass.columns:
        cat_2mass['ext_key'] = ''
    if 'gal_contam' not in cat_2mass.columns:
        cat_2mass['gal_contam'] = 0
    if 'ph_qual' not in cat_2mass.columns:
        cat_2mass['ph_qual'] = 'UUU'
    if 'prox' not in cat_2mass.columns:
        cat_2mass['prox'] = 99.0
    # Mark extended sources: ext_key is non-empty and non-null
    cat_2mass['is_extended'] = cat_2mass['ext_key'].fillna('').astype(str).str.strip().ne('')
    
    if cat_vsx is not None and len(cat_vsx) > 0:
        # Match catalogs
        # Extract numeric values to handle both Quantity and plain arrays (astropy compatibility)
        ra_2mass = np.array(cat_2mass['RAJ2000'], dtype=float)
        dec_2mass = np.array(cat_2mass['DEJ2000'], dtype=float)
        ra_vsx = np.array(cat_vsx['RAJ2000'], dtype=float)
        dec_vsx = np.array(cat_vsx['DEJ2000'], dtype=float)
        
        coords_2mass = SkyCoord(ra=ra_2mass*u.deg, dec=dec_2mass*u.deg)
        coords_vsx = SkyCoord(ra=ra_vsx*u.deg, dec=dec_vsx*u.deg)
        
        idx, sep, _ = coords_2mass.match_to_catalog_sky(coords_vsx)
        matched = sep.arcsec < vsx_tol
        
        cat_2mass.loc[matched, 'variable'] = 1
    
    # Fill missing magnitudes
    for band in ['H', 'J', 'K']:
        mag_col = band
        err_col = f'e{band}'
        
        # Missing magnitude -> 99
        cat_2mass[mag_col] = cat_2mass[mag_col].fillna(default_mag)
        
        # Missing error but has magnitude -> 0.4
        has_mag = cat_2mass[mag_col] != default_mag
        cat_2mass.loc[has_mag, err_col] = cat_2mass.loc[has_mag, err_col].fillna(default_err)
        
        # Missing magnitude -> error also 99
        cat_2mass.loc[~has_mag, err_col] = default_mag
    
    return cat_2mass

def group_coadds_by_position(coadd_files, config, verbose=False):
    """Group coadd files by RA/DEC within tolerance.
    
    Images within grouping_tolerance_arcmin share the same catalog download.
    The catalog is then cut per-image to sources within image bounds + margin.
    """
    if verbose:
        logging.info("Grouping coadds by position...")
    
    tolerance_arcmin = config['astrometry']['catalog'].get('grouping_tolerance_arcmin', 1.0)
    
    # Extract coordinates
    coords_data = []
    for coadd_path in coadd_files:
        with fits.open(coadd_path) as hdul:
            header = hdul[0].header
            ra = header.get('RA', 0)
            dec = header.get('DEC', 0)
            coords_data.append({
                'file': coadd_path,
                'ra': ra,
                'dec': dec,
                'coord': SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
            })
    
    # Group by proximity
    groups = []
    used = set()
    
    for i, data_i in enumerate(coords_data):
        if i in used:
            continue
        
        group = [data_i]
        used.add(i)
        
        for j, data_j in enumerate(coords_data):
            if j in used:
                continue
            
            sep = data_i['coord'].separation(data_j['coord']).arcmin
            if sep < tolerance_arcmin:
                group.append(data_j)
                used.add(j)
        
        # Calculate mean position
        mean_ra = np.mean([d['ra'] for d in group])
        mean_dec = np.mean([d['dec'] for d in group])
        
        groups.append({
            'ra': mean_ra,
            'dec': mean_dec,
            'files': [d['file'] for d in group]
        })
    
    return groups

def download_catalogs_for_groups(position_groups, config, catalog_dir, verbose=False):
    """Download catalogs for each position group. Reuses cached files if present."""
    if verbose:
        logging.info("Downloading catalogs...")
    
    os.makedirs(catalog_dir, exist_ok=True)
    
    radius_arcmin = config['astrometry']['catalog']['radius_arcmin']
    
    catalog_map = {}
    
    for i, group in enumerate(position_groups):
        ra = group['ra']
        dec = group['dec']
        
        if verbose:
            logging.info(f"  Position {i+1}: RA={ra:.4f}, DEC={dec:.4f}")
        
        cat_filename = f"catalog_{i+1}_ra{ra:.4f}_dec{dec:.4f}.csv"
        cat_path = os.path.join(catalog_dir, cat_filename)
        
        # Reuse cached catalog if it exists and has data
        if os.path.isfile(cat_path) and os.path.getsize(cat_path) > 100:
            if verbose:
                logging.info(f"    Using cached catalog: {cat_filename}")
            for coadd_file in group['files']:
                catalog_map[coadd_file] = cat_path
            continue
        
        # Download 2MASS
        cat_2mass = download_catalog(ra, dec, radius_arcmin, '2mass', config)
        
        # Download VSX
        cat_vsx = download_catalog(ra, dec, radius_arcmin, 'vsx', config)
        
        if cat_2mass is None:
            logging.error(f"  Failed to download 2MASS catalog for position {i+1}")
            continue
        
        # Process catalog
        cat_processed = process_catalog(cat_2mass, cat_vsx, config)
        
        # Save relevant columns (ext_key for star/galaxy separation, gal_contam for contamination flag)
        cols = ['RAJ2000', 'DEJ2000', 'H', 'J', 'K', 'eH', 'eJ', 'eK',
                'variable', 'ext_key', 'gal_contam', 'is_extended',
                'ph_qual', 'prox']
        cols = [c for c in cols if c in cat_processed.columns]
        cat_processed[cols].to_csv(cat_path, index=False)
        
        # Map catalog to files
        for coadd_file in group['files']:
            catalog_map[coadd_file] = cat_path
    
    return catalog_map

# ============================================================================
# STEP 5: ASTROMETRY
# ============================================================================

def detect_sources(data, config, npix_min, threshold_sigma, return_rms_matrix=False):
    """Detect sources using SEP with iterative background refinement.
    
    For crowded fields, iteratively masks detected sources and re-estimates
    the background to avoid over-subtraction.
    
    Parameters:
        data: two-dimensional image array
        config: Configuration dictionary
        npix_min: Minimum connected pixels above threshold
        threshold_sigma: Detection threshold in sigma
        return_rms_matrix: If True, also return two-dimensional background RMS map
    
    Returns:
        sources: List of dicts with 'x', 'y', 'flux', 'mag' keys
        data_sub: Background-subtracted data
        bkg_rms: Background RMS (scalar)
        bkg_rms_matrix: two-dimensional background RMS map (only if return_rms_matrix=True)
    """
    # Ensure native byte order and proper dtype
    if not data.dtype.isnative:
        data = data.astype(data.dtype.newbyteorder('='))
    data = np.ascontiguousarray(data.astype(np.float32))
    
    # Get detection parameters from config
    det_cfg = config['detection']
    bw = det_cfg.get('sep_bw', 128)
    bh = det_cfg.get('sep_bh', 128)
    fw = det_cfg.get('sep_fw', 2)
    fh = det_cfg.get('sep_fh', 2)
    n_iter = det_cfg.get('n_iter', 2)
    mask_scale = det_cfg.get('iter_mask_scale', 2.5)
    deblend_nthresh = det_cfg.get('deblend_nthresh', 64)
    deblend_cont = det_cfg.get('deblend_cont', 0.005)
    
    # Initialize mask with invalid pixels
    mask = np.zeros(data.shape, dtype=bool)
    mask |= ~np.isfinite(data)
    
    inst_zp = det_cfg.get('instrumental_zeropoint', 25.0)
    default_mag = config['astrometry']['catalog'].get('default_mag', 99.0)
    
    objects = None
    
    for iteration in range(n_iter):
        # Estimate background with current mask
        try:
            bkg = sep.Background(data, bw=bw, bh=bh, fw=fw, fh=fh, mask=mask)
        except Exception:
            # Fallback without mask if it fails
            bkg = sep.Background(data, bw=bw, bh=bh, fw=fw, fh=fh)
        
        data_sub = data - bkg.back()
        bkg_rms = bkg.globalrms
        threshold = threshold_sigma * bkg_rms
        
        # Raise deblending sub-object limit to avoid overflow on crowded fields
        try:
            sep.set_sub_object_limit(4096)
        except Exception:
            pass
        
        # Extract sources
        objects = sep.extract(data_sub, threshold, minarea=npix_min,
                              deblend_nthresh=deblend_nthresh,
                              deblend_cont=deblend_cont)
        
        # Update mask with detected sources for next iteration (except last)
        if iteration < n_iter - 1 and len(objects) > 0:
            for obj in objects:
                try:
                    sep.mask_ellipse(mask, obj['x'], obj['y'],
                                     obj['a'], obj['b'], obj['theta'],
                                     r=mask_scale)
                except Exception:
                    # Fallback to circular mask
                    cy, cx = int(round(obj['y'])), int(round(obj['x']))
                    radius = int(round(mask_scale * max(obj['a'], obj['b'], 3)))
                    y, x = np.ogrid[:data.shape[0], :data.shape[1]]
                    circ_mask = (x - cx)**2 + (y - cy)**2 <= radius**2
                    mask |= circ_mask
    
    # Residual re-extraction: subtract detected source models and re-detect
    # to find faint sources hidden under the wings of bright neighbors
    if objects is not None and len(objects) > 0:
        residual = data_sub.copy()
        ny, nx = residual.shape
        for obj in objects:
            # Subtract each source as a 2D Gaussian matching SEP parameters
            cx, cy = obj['x'], obj['y']
            a_s, b_s = max(obj['a'], 0.5), max(obj['b'], 0.5)
            theta = obj['theta']
            peak = obj['peak']
            r_cut = int(np.ceil(max(a_s, b_s) * 5))
            x0, x1 = max(0, int(cx) - r_cut), min(nx, int(cx) + r_cut + 1)
            y0, y1 = max(0, int(cy) - r_cut), min(ny, int(cy) + r_cut + 1)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            dx = xx - cx
            dy = yy - cy
            xr = cos_t * dx + sin_t * dy
            yr = -sin_t * dx + cos_t * dy
            model = peak * np.exp(-0.5 * (xr**2 / a_s**2 + yr**2 / b_s**2))
            residual[y0:y1, x0:x1] -= model
        
        # Re-extract on the residual image
        try:
            new_objects = sep.extract(residual, threshold, minarea=npix_min,
                                      deblend_nthresh=deblend_nthresh,
                                      deblend_cont=deblend_cont)
            if len(new_objects) > 0:
                # Keep only genuinely new detections (not near existing ones)
                existing_xy = np.column_stack([objects['x'], objects['y']])
                new_xy = np.column_stack([new_objects['x'], new_objects['y']])
                tree = cKDTree(existing_xy)
                dists, _ = tree.query(new_xy)
                min_sep = max(2.0, np.median(np.sqrt(objects['a']**2 + objects['b']**2)) * 1.5)
                novel = dists > min_sep
                if np.any(novel):
                    novel_objs = new_objects[novel]
                    # Re-measure flux on original (un-subtracted) data_sub
                    for i in range(len(novel_objs)):
                        ox, oy = novel_objs['x'][i], novel_objs['y'][i]
                        r_ap = max(novel_objs['a'][i], novel_objs['b'][i]) * 2.5
                        flux_sum, _, _ = sep.sum_circle(data_sub, [ox], [oy], r_ap)
                        novel_objs['flux'][i] = flux_sum[0]
                        # Re-measure peak from original image
                        px, py = int(round(ox)), int(round(oy))
                        py0, py1 = max(0, py-1), min(ny, py+2)
                        px0, px1 = max(0, px-1), min(nx, px+2)
                        novel_objs['peak'][i] = np.max(data_sub[py0:py1, px0:px1])
                    objects = rfn.stack_arrays([objects, novel_objs], usemask=False)
        except Exception:
            pass
    
    # Create sources list (vectorized)
    sources = []
    if objects is not None and len(objects) > 0:
        # Refine centroids using windowed algorithm (iterative Gaussian window)
        # This is equivalent to SExtractor's XWIN_IMAGE/YWIN_IMAGE and is
        # significantly more precise than isophotal centroids for undersampled PSFs.
        # sig = geometric mean of semi-axes (proxy for FWHM/2.35)
        x_iso = objects['x'].astype(np.float64)
        y_iso = objects['y'].astype(np.float64)
        sig = np.sqrt(objects['a'].astype(np.float64) * objects['b'].astype(np.float64))
        sig = np.maximum(sig, 0.5)  # minimum 0.5 px (SEP default floor)
        try:
            x_win, y_win, win_flag = sep.winpos(data_sub, x_iso, y_iso, sig)
            # Use windowed position only where it converged (flag == 0)
            good = win_flag == 0
            x_use = np.where(good, x_win, x_iso)
            y_use = np.where(good, y_win, y_iso)
        except Exception:
            # Fallback to isophotal if winpos fails entirely
            x_use = x_iso
            y_use = y_iso
        
        # Vectorized magnitude calculation
        fluxes = objects['flux'].astype(np.float64)
        valid_flux = fluxes > 0
        mags = np.full(len(fluxes), default_mag)
        mags[valid_flux] = -2.5 * np.log10(fluxes[valid_flux]) + inst_zp
        
        # Build sources list with refined centroids
        sources = [
            {
                'x': float(x_use[i]),
                'y': float(y_use[i]),
                'flux': float(obj['flux']),
                'mag': float(mags[i]),
                'a': float(obj['a']),
                'b': float(obj['b']),
                'theta': float(obj['theta']),
                'npix': int(obj['npix']),
                'flag': int(obj['flag']),
                'peak': float(obj['peak'])
            }
            for i, obj in enumerate(objects)
        ]
    
    # Sort by magnitude (brightest first)
    sources.sort(key=lambda s: s['mag'])
    
    if return_rms_matrix:
        return sources, data_sub, bkg_rms, bkg.rms()
    return sources, data_sub, bkg_rms

def photometry(sources, data, aperture_radius):
    """Perform aperture photometry."""
    positions = [(s['x'], s['y']) for s in sources]
    apertures = CircularAperture(positions, r=aperture_radius)
    phot_table = aperture_photometry(data, apertures)
    
    for i, source in enumerate(sources):
        source['aperture_flux'] = phot_table['aperture_sum'][i]
    
    return sources

def geometric_hash_code(quad):
    """Calculate geometric hash code for quad using normalized coordinates.

    This matches the notebook implementation which expects `quad` to be an
    iterable of rows with fields 'X' and 'Y'.
    """
    coords = np.array([[row['X'], row['Y']] for row in quad])
    dists = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
    i, j = np.unravel_index(np.argmax(dists), dists.shape)
    A, B = coords[i], coords[j]
    others = [coords[k] for k in range(4) if k not in (i, j)]
    C, D = others
    vec = B - A
    norm2 = np.sum(vec**2)
    perp = np.array([-vec[1], vec[0]])
    xC = np.dot(C - A, vec) / norm2
    yC = np.dot(C - A, perp) / norm2
    xD = np.dot(D - A, vec) / norm2
    yD = np.dot(D - A, perp) / norm2
    if xC > xD:
        xC, xD = xD, xC
        yC, yD = yD, yC
    if xC + xD > 1.0:
        xC, xD = 1 - xC, 1 - xD
        yC, yD = 1 - yC, 1 - yD
    return np.array([xC, yC, xD, yD])

def build_quads_heap(table, G=2000):
    """Build geometric quads from brightest sources using heap selection.
    
    Creates all possible 4-star combinations (quads) from the input table
    and keeps only the G brightest quads (by sum of magnitudes). Uses a
    min-heap for efficient selection without sorting all combinations.
    
    Each quad is represented by its geometric hash code, which encodes the
    relative positions of the 4 stars in a scale/rotation-invariant way.
    
    Args:
        table: Astropy Table with columns 'X', 'Y', 'MAG'
        G: Maximum number of quads to keep (brightest)
    
    Returns:
        list: Dicts with 'quad' (rows), 'hash' (geometric code), 'coords' (xy array)
    """
    table.sort('MAG')
    heap = []
    counter = itertools.count()
    for quad in itertools.combinations(table, 4):
        sum_mag = sum(row['MAG'] for row in quad)
        cnt = next(counter)
        if len(heap) < G:
            heapq.heappush(heap, (-sum_mag, cnt, quad))
        else:
            if sum_mag < -heap[0][0]:
                heapq.heapreplace(heap, (-sum_mag, cnt, quad))
    best_quads = [quad for _, _, quad in sorted(heap, key=lambda x: -x[0])]
    index = [
        {'quad': quad, 'hash': geometric_hash_code(quad),
         'coords': np.array([[row['X'], row['Y']] for row in quad])}
        for quad in best_quads
    ]
    return index

def match_quads(det_quads, cat_quads, config):
    """Find matching quads between detections and catalog.
    
    VECTORIZED implementation - compares all quad pairs using matrix operations.
    Uses scipy.spatial.distance.cdist for fast pairwise L2 distance computation.
    
    Args:
        det_quads: List of detection quads from build_quads_heap()
        cat_quads: List of catalog quads from build_quads_heap()
        config: Configuration with astrometry.threshold_code parameter
    
    Returns:
        list: Matched quad pairs sorted by hash similarity (best first),
              each dict with 'det_quad', 'cat_quad', 'hash_diff'
    """
    threshold = config['astrometry']['threshold_code']
    
    if not det_quads or not cat_quads:
        return []
    
    # Extract hash arrays for vectorized comparison
    det_hashes = np.array([q['hash'] for q in det_quads])
    cat_hashes = np.array([q['hash'] for q in cat_quads])
    
    # Compute all pairwise L2 distances at once
    dists = cdist(det_hashes, cat_hashes, metric='euclidean')
    
    # Find matches below threshold
    match_i, match_j = np.where(dists < threshold)
    match_dists = dists[match_i, match_j]
    
    # Sort by distance (best matches first)
    sort_idx = np.argsort(match_dists)
    
    matches = []
    for idx in sort_idx:
        i, j = match_i[idx], match_j[idx]
        matches.append({
            'det_quad': det_quads[i],
            'cat_quad': cat_quads[j],
            'hash_diff': match_dists[idx]
        })
    
    return matches

def compute_similarity_transform(cat_coords, det_coords):
    """Compute similarity transformation from catalog to detector coordinates.
    
    Finds the best-fit transformation that maps catalog positions to
    detector positions using least-squares fitting. The similarity
    transform preserves angles and relative distances (allows scale,
    rotation, and translation).
    
    Uses singular value decomposition (SVD) for robust fitting:
    1. Center both coordinate sets
    2. Compute optimal rotation via SVD
    3. Compute scale from variance ratio
    4. Compute translation from centroids
    
    Args:
        cat_coords: Nx2 array of catalog (x, y) positions
        det_coords: Nx2 array of detector (x, y) positions
    
    Returns:
        tuple: (scale, R, t) where scale is the uniform scale factor,
               R is the rotation matrix, and t is the translation vector.
    """
    cat_coords = np.asarray(cat_coords, dtype=np.float64)
    det_coords = np.asarray(det_coords, dtype=np.float64)
    
    cat_mean = np.mean(cat_coords, axis=0)
    det_mean = np.mean(det_coords, axis=0)
    
    X = cat_coords - cat_mean
    Y = det_coords - det_mean
    
    C = X.T @ Y
    U, S, Vt = np.linalg.svd(C)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T
    
    scale = np.trace(R @ C) / np.sum(X**2)
    t = det_mean - scale * (R @ cat_mean)
    
    return scale, R, t


# ============================================================================
# STEP 6: PHOTOMETRIC CALIBRATION
# ============================================================================

def detect_sources_for_photometry(data, config):
    """Detect sources for photometry using fixed parameters from config.
    
    Wrapper around detect_sources() with photometry-specific parameters.
    
    Returns:
        sources: List of dicts with 'x', 'y', 'flux', 'mag' keys
        data_sub: Background-subtracted data
        bkg_rms: Background RMS (scalar)
        bkg_rms_matrix: Background RMS map (same shape as image)
    """
    phot_cfg = config['photometry']
    return detect_sources(
        data, config,
        npix_min=phot_cfg['min_pixels'],
        threshold_sigma=phot_cfg['threshold_sigma'],
        return_rms_matrix=True
    )


def get_isolation_mask(positions, min_distance=7.0):
    """Return mask for isolated sources (no neighbors within min_distance pixels)."""
    tree = cKDTree(positions)
    
    # Query for neighbors within min_distance (k=2 to get self + nearest)
    distances, _ = tree.query(positions, k=2, distance_upper_bound=min_distance)
    
    # Isolated = no second neighbor within min_distance
    isolated_mask = distances[:, 1] >= min_distance
    
    return isolated_mask


def get_central_mask(positions, image_shape, central_fraction=0.90):
    """Return mask for sources in central region of image."""
    ny, nx = image_shape
    
    # Calculate central region boundaries
    border_x = (1.0 - central_fraction) / 2.0 * nx
    border_y = (1.0 - central_fraction) / 2.0 * ny
    
    x_min, x_max = border_x, nx - border_x
    y_min, y_max = border_y, ny - border_y
    
    x = positions[:, 0]
    y = positions[:, 1]
    
    central_mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    
    return central_mask


def get_stellar_mask(sources, config):
    """Return mask selecting only point-source-like detections.
    
    Identifies the stellar locus from the population of detected sources
    and rejects objects whose size or ellipticity deviate significantly.
    Point sources share the same PSF, so their semi-axes cluster tightly;
    galaxies, blends, and artifacts are outliers.
    
    Criteria:
    1. SEP extraction flag == 0 (clean extraction, no blending/truncation)
    2. Size (geometric mean of semi-axes sqrt(a*b)) within n_sigma of the
       clipped median — rejects extended objects and cosmics/hot pixels
    3. Ellipticity (1 - b/a) below a maximum threshold — rejects elongated
       artifacts, diffraction spikes, and merging doubles
    """
    morph_cfg = config['photometry'].get('star_selection', {})
    max_ellipticity = morph_cfg.get('max_ellipticity', 0.4)
    size_sigma = morph_cfg.get('size_sigma', 3.0)
    max_flag = morph_cfg.get('max_flag', 0)
    
    n = len(sources)
    if n < 5:
        return np.ones(n, dtype=bool)
    
    a_arr = np.array([s['a'] for s in sources])
    b_arr = np.array([s['b'] for s in sources])
    flags = np.array([s.get('flag', 0) for s in sources])
    
    # Geometric mean of semi-axes = proxy for PSF sigma
    size = np.sqrt(a_arr * b_arr)
    ellipticity = 1.0 - np.clip(b_arr / np.maximum(a_arr, 1e-6), 0, 1)
    
    # Flag filter: keep only cleanly-extracted sources
    flag_ok = flags <= max_flag
    
    # Ellipticity filter
    ell_ok = ellipticity <= max_ellipticity
    
    # Size filter: sigma-clipped median of the stellar locus
    # Use only round, unflagged sources to define the locus
    locus_mask = flag_ok & ell_ok
    if np.sum(locus_mask) < 5:
        locus_mask = np.ones(n, dtype=bool)
    
    locus_sizes = size[locus_mask]
    med_size = np.median(locus_sizes)
    mad_size = np.median(np.abs(locus_sizes - med_size))
    sigma_size = 1.4826 * mad_size  # MAD → Gaussian σ
    sigma_size = max(sigma_size, 0.1)  # floor to avoid zero-width locus
    
    size_ok = np.abs(size - med_size) < size_sigma * sigma_size
    
    stellar_mask = flag_ok & ell_ok & size_ok
    return stellar_mask


def aperture_photometry_with_noise(positions, data_sub, noise_matrix, aperture_radius):
    """Perform aperture photometry computing errors from noise matrix.
    
    Parameters:
    -----------
    positions : array of (x, y) positions
    data_sub : background-subtracted data
    noise_matrix : two-dimensional noise/RMS map
    aperture_radius : aperture radius in pixels
    
    Returns:
    --------
    fluxes : array of aperture fluxes
    flux_errors : array of flux errors from noise matrix
    """
    apertures = CircularAperture(positions, r=aperture_radius)
    
    # Photometry on data
    phot_data = aperture_photometry(data_sub, apertures)
    fluxes = np.array(phot_data['aperture_sum'])
    
    # Photometry on variance (noise^2) to get flux variance
    variance_matrix = noise_matrix ** 2
    phot_noise = aperture_photometry(variance_matrix, apertures)
    flux_errors = np.sqrt(np.array(phot_noise['aperture_sum']))
    
    return fluxes, flux_errors


def compute_limiting_magnitude(zp, noise_map, aperture_radius, inst_zp=25.0,
                               target_snr=3.0, central_fraction=0.8):
    """Compute limiting magnitude from the image noise properties.
    
    Instead of extrapolating from detected sources (unreliable with few stars),
    this method estimates the noise in a typical empty-sky aperture directly from
    the noise map, then converts the flux at the target S/N to a calibrated
    magnitude.
    
    Algorithm:
    1. Estimate the typical per-pixel noise from the central region of the
       noise map (median, avoiding edges and masked pixels)
    2. Compute the noise in a circular aperture: σ_aper = σ_pix × √(N_pix)
       where N_pix = π × r²
    3. The faintest detectable flux at target S/N: F_lim = target_snr × σ_aper
    4. Convert to calibrated magnitude: mag_lim = (inst_zp - 2.5·log10(F_lim)) + zp
    
    Args:
        zp: Photometric zero point (calibrated - instrumental)
        noise_map: 2D noise (RMS) map combining all noise sources
                   (Poisson + read noise + sky subtraction + background estimation)
        aperture_radius: Aperture radius in pixels
        inst_zp: Instrumental zero point (default 25.0)
        target_snr: Signal-to-noise ratio defining the limit (default 3.0 ≈ 0.36 mag error)
        central_fraction: Fraction of image used for noise estimation (default 0.8)
    
    Returns:
        mag_lim: Limiting magnitude, or 99.0 if computation fails
    """
    if noise_map is None or zp is None:
        return 99.0
    
    try:
        ny, nx = noise_map.shape
        
        # Use central region to avoid noisy edges / vignetted corners
        margin_y = int(ny * (1 - central_fraction) / 2)
        margin_x = int(nx * (1 - central_fraction) / 2)
        central = noise_map[margin_y:ny - margin_y, margin_x:nx - margin_x]
        
        # Median per-pixel noise (robust to outliers / source residuals)
        sigma_pixel = np.nanmedian(central)
        
        if sigma_pixel <= 0 or not np.isfinite(sigma_pixel):
            return 99.0
        
        # Noise in a circular aperture (assuming uncorrelated pixels)
        n_pix = np.pi * aperture_radius ** 2
        sigma_aperture = sigma_pixel * np.sqrt(n_pix)
        
        # Flux at the detection limit
        f_lim = target_snr * sigma_aperture
        
        if f_lim <= 0:
            return 99.0
        
        # Calibrated limiting magnitude
        mag_lim = inst_zp - 2.5 * np.log10(f_lim) + zp
        
        return float(mag_lim)
        
    except Exception:
        return 99.0


def match_detections_to_catalog(det_positions, cat_positions, tolerance):
    """Match detections to catalog by nearest neighbour within tolerance.
    
    Ensures uniqueness: each catalog source is matched to at most one detection.
    """
    tree = cKDTree(cat_positions)
    distances, indices = tree.query(det_positions, k=1)
    valid_mask = distances < tolerance
    
    matched_pairs = {}
    for det_idx in np.where(valid_mask)[0]:
        cat_idx = indices[det_idx]
        dist = distances[det_idx]
        if cat_idx not in matched_pairs or dist < matched_pairs[cat_idx][1]:
            matched_pairs[cat_idx] = (det_idx, dist)
    
    matched_det_idx = [pair[0] for pair in matched_pairs.values()]
    matched_cat_idx = list(matched_pairs.keys())
    return np.array(matched_det_idx), np.array(matched_cat_idx)


def fit_zeropoint(mag_inst, mag_cat, mag_inst_err, mag_cat_err,
                  target_rms=None, min_stars=3, verbose=False):
    """Fit zeropoint with iterative worst-outlier rejection.
    
    m_cat = m_inst + ZP  (weighted mean).
    
    Removes one worst outlier per iteration until target_rms is reached
    or min_stars remain.
    
    Returns
    -------
    zp, zp_err, rms, mask, n_used, n_total
    Returns None-filled tuple on failure.
    """
    n_total = len(mag_inst)
    mask = np.ones(n_total, dtype=bool)
    
    iteration = 0
    while True:
        mi = mag_inst[mask]
        mc = mag_cat[mask]
        ei = mag_inst_err[mask]
        ec = mag_cat_err[mask]
        err_combined = np.sqrt(ec**2 + ei**2)
        weights = 1.0 / err_combined**2
        
        zps = mc - mi
        zp = np.sum(zps * weights) / np.sum(weights)
        zp_err = 1.0 / np.sqrt(np.sum(weights))
        residuals = mc - (mi + zp)
        
        rms = np.sqrt(np.mean(residuals**2))
        
        if verbose:
            logging.info(f"      Iter {iteration}: N={np.sum(mask)}, RMS={rms:.3f}")
        
        if target_rms is not None and rms < target_rms:
            break
        if np.sum(mask) <= min_stars:
            break
        
        worst_idx = np.argmax(np.abs(residuals))
        mask_indices = np.where(mask)[0]
        mask[mask_indices[worst_idx]] = False
        iteration += 1
    
    n_used = int(np.sum(mask))
    return zp, zp_err, rms, mask, n_used, n_total


def generate_photometry_plot(mag_inst_all, mag_cat_all, mag_inst_err_all, 
                            mag_cat_err_all, final_mask, zeropoint, zeropoint_err, rms, 
                            n_used, filter_name, filename, output_path):
    """Generate calibration plot with zeropoint fit and residuals.
    
    Shows used stars in blue and rejected stars as red crosses.
    """
    
    # Split into used and rejected stars
    mag_inst_used = mag_inst_all[final_mask]
    mag_cat_used = mag_cat_all[final_mask]
    mag_inst_err_used = mag_inst_err_all[final_mask]
    mag_cat_err_used = mag_cat_err_all[final_mask]
    
    mag_inst_rejected = mag_inst_all[~final_mask]
    mag_cat_rejected = mag_cat_all[~final_mask]
    
    residuals_used = mag_cat_used - (mag_inst_used + zeropoint)
    residuals_rejected = mag_cat_rejected - (mag_inst_rejected + zeropoint)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Calibration relation
    ax1 = axes[0]
    
    # Plot rejected stars as red crosses
    if len(mag_inst_rejected) > 0:
        ax1.plot(mag_inst_rejected, mag_cat_rejected, 'rx', markersize=8, 
                markeredgewidth=2, alpha=0.6, label=f'Rejected ({len(mag_inst_rejected)})')
    
    # Plot used stars in blue
    ax1.errorbar(mag_inst_used, mag_cat_used, 
                 yerr=mag_cat_err_used, xerr=mag_inst_err_used,
                 fmt='o', alpha=0.6, markersize=6, color='blue', label=f'Used ({n_used})')
    
    # Fit line in green
    x_fit = np.linspace(mag_inst_all.min(), mag_inst_all.max(), 100)
    y_fit = x_fit + zeropoint
    ax1.plot(x_fit, y_fit, 'g-', linewidth=2, 
             label=f'ZP = {zeropoint:.3f} ± {zeropoint_err:.3f}')
    
    ax1.set_xlabel('Instrumental Magnitude', fontsize=12, fontweight='bold')
    ax1.set_ylabel(f'Catalog Magnitude (2MASS {filter_name})', fontsize=12, fontweight='bold')
    ax1.set_title(f'Photometric Calibration', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()
    ax1.invert_yaxis()
    
    # Plot 2: Residuals
    ax2 = axes[1]
    
    # Plot rejected stars as red crosses
    if len(mag_cat_rejected) > 0:
        ax2.plot(mag_cat_rejected, residuals_rejected, 'rx', markersize=8, 
                markeredgewidth=2, alpha=0.6)
    
    # Plot used stars in blue
    ax2.errorbar(mag_cat_used, residuals_used, yerr=mag_inst_err_used,
                 fmt='o', alpha=0.6, markersize=6, color='blue')
    ax2.axhline(0, color='g', linestyle='--', linewidth=2, label='Zero')
    ax2.axhline(rms, color='gray', linestyle=':', linewidth=2, label=f'±{rms:.3f} mag')
    ax2.axhline(-rms, color='gray', linestyle=':', linewidth=2)
    
    ax2.set_xlabel(f'Catalog Magnitude (2MASS {filter_name})', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Residuals (mag)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Residuals (RMS = {rms:.3f} mag)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()
    
    # Add filename as super title
    fig.suptitle(filename, fontsize=10, y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def generate_detection_map(data, sources, aperture_radius, filename, output_path, config,
                           targets=None, wcs=None, sources_data=None, mag_lim=None,
                           pixel_scale=None):
    """Save a PNG showing the image with all detected sources circled.
    
    If targets are provided (from -f file), draws an X at each target position
    with an error circle. If a detected source falls within the error radius,
    highlights it in green and labels it with its calibrated magnitude.
    Otherwise labels it with the limiting magnitude.
    
    Parameters are read from config['detection_map'].
    """
    dm_cfg = config.get('detection_map', {})
    central_fraction = dm_cfg.get('central_fraction', 0.8)
    pct_lo = dm_cfg.get('percentile_low', 5)
    pct_hi = dm_cfg.get('percentile_high', 95)
    cmap = dm_cfg.get('colormap', 'Greys')
    dpi = dm_cfg.get('dpi', 200)
    figsize = dm_cfg.get('figsize', [8, 8])
    circ_color = dm_cfg.get('circle_color', 'red')
    circ_lw = dm_cfg.get('circle_linewidth', 0.6)
    circ_alpha = dm_cfg.get('circle_alpha', 0.8)
    title_fs = dm_cfg.get('title_fontsize', 10)

    ny, nx = data.shape
    margin = int((1.0 - central_fraction) / 2.0 * ny)
    central = data[margin:ny - margin, margin:nx - margin]
    finite = central[np.isfinite(central)]
    if len(finite) == 0:
        return
    vmin, vmax = np.percentile(finite, [pct_lo, pct_hi])

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
              interpolation='nearest', aspect='equal')

    for s in sources:
        circ = plt.Circle((s['x'], s['y']), aperture_radius,
                           edgecolor=circ_color, facecolor='none',
                           linewidth=circ_lw, alpha=circ_alpha)
        ax.add_patch(circ)

    ax.set_title(f"{filename}  —  {len(sources)} detections", fontsize=title_fs)
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)
    ax.tick_params(labelsize=8)
    
    # Overlay target positions from -f file
    if targets and wcs is not None and pixel_scale:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        
        for tgt in targets:
            # Convert target RA/Dec to pixel coordinates
            try:
                tgt_x, tgt_y = wcs.all_world2pix(tgt['ra_deg'], tgt['dec_deg'], 0)
            except Exception:
                continue
            
            # Skip if outside image
            if tgt_x < -10 or tgt_x > nx + 10 or tgt_y < -10 or tgt_y > ny + 10:
                continue
            
            # Error circle radius in pixels
            r_px = tgt['radius_arcsec'] / pixel_scale
            
            # Draw X marker at target position
            marker_size = max(r_px * 0.6, 4)
            ax.plot(tgt_x, tgt_y, 'x', color='cyan', markersize=marker_size,
                    markeredgewidth=1.5, zorder=10)
            
            # Draw error circle
            err_circ = plt.Circle((tgt_x, tgt_y), r_px,
                                   edgecolor='cyan', facecolor='none',
                                   linewidth=1.2, linestyle='--', alpha=0.9, zorder=10)
            ax.add_patch(err_circ)
            
            # Search for a detected source within error radius
            matched = False
            forced = False
            if sources_data:
                best_dist = np.inf
                best_mag = None
                best_mag_err = None
                best_x, best_y = None, None
                best_flag = None
                
                for sd in sources_data:
                    tgt_coord = SkyCoord(tgt['ra_deg'], tgt['dec_deg'], unit=u.deg)
                    src_coord = SkyCoord(sd['ra'], sd['dec'], unit=u.deg)
                    sep_arcsec = tgt_coord.separation(src_coord).arcsec
                    
                    if sep_arcsec <= tgt['radius_arcsec'] and sep_arcsec < best_dist:
                        best_dist = sep_arcsec
                        best_mag = sd['mag_cal']
                        best_mag_err = sd.get('e_mag_cal', 99.0)
                        best_x, best_y = sd['x'], sd['y']
                        best_flag = sd.get('flag', 0)
                
                if best_mag is not None and np.isfinite(best_mag) and best_mag < 90:
                    if best_flag == 4:
                        # Forced photometry result
                        forced = True
                        max_forced_err = config['photometry'].get('max_forced_mag_err', 0.33)
                        if best_mag_err < max_forced_err:
                            label = f"F:{best_mag:.2f}±{best_mag_err:.2f}"
                            color = 'yellow'
                        else:
                            label = f">{mag_lim:.2f}" if mag_lim and mag_lim < 90 else "N/D"
                            color = 'orange'
                        # Draw forced aperture circle
                        forced_circ = plt.Circle((best_x, best_y), aperture_radius,
                                                  edgecolor='yellow', facecolor='none',
                                                  linewidth=1.2, linestyle=':', zorder=11)
                        ax.add_patch(forced_circ)
                    else:
                        matched = True
                        # Highlight matched source in green
                        match_circ = plt.Circle((best_x, best_y), aperture_radius,
                                                edgecolor='lime', facecolor='none',
                                                linewidth=1.5, zorder=11)
                        ax.add_patch(match_circ)
                        label = f"{best_mag:.2f}"
                        color = 'lime'
            
            if not matched and not forced:
                label = f">{mag_lim:.2f}" if mag_lim and mag_lim < 90 else "N/D"
                color = 'orange'
            
            ax.annotate(label, (tgt_x, tgt_y), textcoords='offset points',
                        xytext=(8, 8), fontsize=7, color=color, fontweight='bold',
                        zorder=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def save_photometry_catalog(output_path, sources_data, zeropoint, zeropoint_err, 
                           rms, n_stars, aperture_radius, filter_name, filename,
                           config, mag_lim=None, threshold_sigma=None, min_pixels=None, n_total=None,
                           date_obs=None, exptime=None, object_name=None,
                           is_coadd=False, ncoadd=None):
    """Save photometry catalog as .txt file with calibration info in header.
    
    Columns: ra dec x y mag_inst e_mag_inst mag_cat e_mag_cat mag_cal e_mag_cal flag
    -99 -99 for sources not in catalog
    Flag: 0=isolated+central, 1=crowded, 2=border, 3=crowded+border
    """
    # Calculate quality using helper functions
    rms_quality = calculate_rms_quality(rms, config)
    rejection_quality, rejected_fraction = calculate_rejection_quality(n_stars, n_total, config)
    
    with open(output_path, 'w') as f:
        # Write header comments
        f.write(f"# Photometric Calibration for {filename}\n")
        f.write(f"# OBJECT: {object_name}\n")
        if date_obs is not None:
            f.write(f"# DATE-OBS: {date_obs}\n")
        if exptime is not None:
            f.write(f"# EXPTIME: {exptime} s\n")
        if is_coadd:
            f.write(f"# Image type: COADD ({ncoadd} frames)\n")
        else:
            f.write(f"# Image type: SINGLE\n")
        f.write(f"# Filter: {filter_name}\n")
        f.write(f"# Zeropoint: {zeropoint:.4f} +/- {zeropoint_err:.4f} mag\n")
        f.write(f"# RMS residuals: {rms:.4f} mag\n")
        f.write(f"# RMS quality: {rms_quality}\n")
        f.write(f"# Calibration stars: {n_stars}\n")
        if n_total is not None:
            f.write(f"# Stars rejected: {n_total - n_stars}/{n_total} ({rejected_fraction*100:.1f}%)\n")
            f.write(f"# Rejection quality: {rejection_quality}\n")
        f.write(f"# Fixed aperture radius: {aperture_radius:.2f} pixels\n")
        if threshold_sigma is not None:
            f.write(f"# Detection threshold: {threshold_sigma:.1f} sigma\n")
        if min_pixels is not None:
            f.write(f"# Minimum pixels: {min_pixels}\n")
        if mag_lim is not None:
            f.write(f"# MagLim: {mag_lim:.2f}\n")
        f.write("#\n")
        f.write("# Columns: ra dec x y mag_inst e_mag_inst mag_cat e_mag_cat mag_cal e_mag_cal flag\n")
        f.write("# Note: mag_cat and e_mag_cat are -99 -99 for sources not matched to 2MASS catalog\n")
        f.write("# Flag: 0=isolated+central (best), 1=crowded, 2=border, 3=crowded+border, 4=forced\n")
        f.write("#\n")
        
        # Write data
        for src in sources_data:
            line = (f"{src['ra']:.6f} {src['dec']:.6f} "
                   f"{src['x']:.3f} {src['y']:.3f} "
                   f"{src['mag_inst']:.4f} {src['e_mag_inst']:.4f} "
                   f"{src['mag_cat']:.4f} {src['e_mag_cat']:.4f} "
                   f"{src['mag_cal']:.4f} {src['e_mag_cal']:.4f} "
                   f"{src['flag']}\n")
            f.write(line)


def photometric_calibration(fits_path, catalog_path, config, output_dir, verbose=False, targets=None):
    """Perform photometric calibration on an astrometrically calibrated FITS file.
    
    Uses a single fixed aperture radius (from config) for all sources.
    Calibration stars are filtered to keep only isolated, stellar, non-variable
    sources with good 2MASS photometric quality.
    
    Parameters:
    -----------
    fits_path : str
        Path to astrometrically calibrated FITS file
    catalog_path : str
        Path to catalog CSV file with 2MASS format columns
    config : dict
        Configuration dictionary
    output_dir : str
        Output directory for calibration files
    verbose : bool
        Verbose output
    
    Returns:
    --------
    success : bool
        True if calibration succeeded
    calib_info : dict or None
        Calibration info including zeropoint, aperture, etc.
    """
    phot_cfg = config['photometry']
    
    # Check if photometry is enabled
    if not phot_cfg.get('enabled', True):
        return False, None
    
    # Read FITS file
    with fits.open(fits_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        
        # Get noise matrix if available
        if len(hdul) > 1 and hdul[1].name == 'NOISE':
            noise_matrix = hdul[1].data
        else:
            noise_matrix = None  # Will be estimated after handling dimensions
    
    # Handle multi-dimensional data
    if data.ndim > 2:
        if data.ndim == 3:
            data = data[0]
            if noise_matrix is not None and noise_matrix.ndim == 3:
                noise_matrix = noise_matrix[0]
        elif data.ndim == 4:
            data = data[0, 0]
            if noise_matrix is not None and noise_matrix.ndim == 4:
                noise_matrix = noise_matrix[0, 0]
    
    # Estimate noise from data if not available
    if noise_matrix is None:
        gain = config['detector']['gain']
        read_noise = config['detector']['read_noise']
        noise_matrix = np.sqrt(np.clip(data, 0, None) / gain + (read_noise / gain) ** 2)
    
    # Get WCS and filter
    wcs = WCS(header)
    filter_name = header.get('FILTER', 'H').strip().upper()
    filename = os.path.basename(fits_path)
    exptime = header.get('EXPTIME', 0.0)
    
    # Determine if this is a coadd or single frame
    dithid = header.get('DITHID', -1)
    pstatsub = header.get('PSTATSUB', -1)
    dithid_coadd = config['fits_markers']['dithid_coadd']
    pstatsub_coadd = config['fits_markers']['pstatsub_coadd']
    is_coadd = (dithid == dithid_coadd) or (pstatsub == pstatsub_coadd)
    ncoadd = header.get('NCOADD', None)
    
    # Check if filter should be calibrated
    calibrate_filters = phot_cfg.get('calibrate_filters', ['J', 'H', 'K'])
    if filter_name not in calibrate_filters:
        if verbose:
            logging.info(f"    Skipping photometry for filter {filter_name} (not in {calibrate_filters})")
        return False, None
    
    if verbose:
        logging.info(f"    Photometric calibration for {filename} (filter {filter_name})")
    
    # Detect sources and get SEP background-subtracted data
    sources, data_sub, bkg_rms, bkg_rms_matrix = detect_sources_for_photometry(data, config)
    
    # Combine FITS noise with SEP background RMS for proper photometric errors
    # FITS noise includes: Poisson (source + sky) + read noise + sky subtraction noise
    # SEP bkg RMS represents uncertainty in local background estimation
    # When SEP subtracts its background mesh, this uncertainty propagates to flux errors
    # Total noise = sqrt(FITS_noise² + SEP_bkg_rms²)
    photometry_noise = np.sqrt(noise_matrix**2 + bkg_rms_matrix**2)
    
    if len(sources) < phot_cfg['min_calibration_stars']:
        if verbose:
            logging.warning(f"    Too few sources detected: {len(sources)}")
        return False, None
    
    # Get source positions
    positions = np.array([[s['x'], s['y']] for s in sources])
    
    # Apply isolation + central masks for calibration
    central_fraction = phot_cfg.get('central_fraction', 0.90)
    min_isolation = phot_cfg.get('min_isolation_dist', 7.0)
    
    isolation_mask = get_isolation_mask(positions, min_isolation)
    central_mask = get_central_mask(positions, data.shape, central_fraction)
    stellar_mask = get_stellar_mask(sources, config)
    calib_mask = isolation_mask & central_mask & stellar_mask
    
    n_calib_sources = np.sum(calib_mask)
    n_stellar = np.sum(stellar_mask)
    if verbose:
        logging.info(f"    Detected {len(sources)} sources, "
                     f"{n_stellar} stellar, "
                     f"{n_calib_sources} stellar+isolated+central for calibration")
    
    if n_calib_sources < phot_cfg['min_calibration_stars']:
        if verbose:
            logging.warning(f"    Too few isolated+central sources: {n_calib_sources}")
        return False, None
    
    # Load catalog and filter by band and variable stars
    catalog_df = pd.read_csv(catalog_path)
    default_mag = config['astrometry']['catalog']['default_mag']
    
    # Filter: has valid magnitude in this band, not variable, and not extended (galaxy)
    valid_cat = (catalog_df[filter_name] != default_mag) & (catalog_df['variable'] == 0)
    # Exclude extended sources (galaxies) using 2MASS ext_key / is_extended flag
    if 'is_extended' in catalog_df.columns:
        valid_cat = valid_cat & (~catalog_df['is_extended'].astype(bool))
    elif 'ext_key' in catalog_df.columns:
        valid_cat = valid_cat & (catalog_df['ext_key'].fillna('').astype(str).str.strip() == '')
    # Exclude galaxy-contaminated sources
    if 'gal_contam' in catalog_df.columns:
        valid_cat = valid_cat & (catalog_df['gal_contam'].fillna(0).astype(int) == 0)
    # Require good 2MASS photometric quality in the relevant band
    # ph_qual is a 3-char string (J,H,K); A or B = reliable for calibration
    if 'ph_qual' in catalog_df.columns:
        band_idx = {'J': 0, 'H': 1, 'K': 2}.get(filter_name, None)
        if band_idx is not None:
            pq = catalog_df['ph_qual'].fillna('UUU').astype(str)
            valid_cat = valid_cat & pq.str[band_idx].isin(['A', 'B'])
    # Require 2MASS isolation: prox > min_isolation_dist × pixel_scale
    # (source not blended in 2MASS itself)
    if 'prox' in catalog_df.columns:
        min_prox_arcsec = phot_cfg.get('min_isolation_dist', 6.0) * config['detector']['pixel_scale']
        valid_cat = valid_cat & (catalog_df['prox'].fillna(0).astype(float) > min_prox_arcsec)
    catalog_filtered = catalog_df[valid_cat].copy()
    
    if len(catalog_filtered) < phot_cfg['min_calibration_stars']:
        if verbose:
            logging.warning(f"    Too few non-variable stellar catalog sources: {len(catalog_filtered)}")
        return False, None
    
    # Convert catalog to pixel coordinates
    cat_ra = catalog_filtered['RAJ2000'].values
    cat_dec = catalog_filtered['DEJ2000'].values
    cat_x, cat_y = wcs.all_world2pix(cat_ra, cat_dec, 0)
    cat_positions = np.column_stack([cat_x, cat_y])
    
    # Filter catalog to sources within image
    ny, nx = data.shape
    in_image = (cat_x >= 0) & (cat_x < nx) & (cat_y >= 0) & (cat_y < ny)
    cat_positions = cat_positions[in_image]
    catalog_filtered = catalog_filtered[in_image].reset_index(drop=True)
    
    if verbose:
        logging.info(f"    Catalog: {len(catalog_filtered)} non-variable stellar sources in image")
    
    # Config parameters
    fixed_aperture = phot_cfg.get('aperture_radius', 3.0)
    match_tolerance = phot_cfg.get('match_tolerance', 2.0)
    default_cat_err = config['astrometry']['catalog'].get('default_error', 0.4)
    inst_zp = config['detection'].get('instrumental_zeropoint', 25.0)
    max_inst_err = phot_cfg.get('max_inst_mag_err', 0.2)
    
    # Match detections to catalog
    matched_det_idx, matched_cat_idx = match_detections_to_catalog(
        positions, cat_positions, match_tolerance
    )
    
    if len(matched_det_idx) < phot_cfg['min_calibration_stars']:
        if verbose:
            logging.warning(f"    Too few matches: {len(matched_det_idx)}")
        return False, None
    
    if verbose:
        logging.info(f"    Matched {len(matched_det_idx)} sources to catalog")
    
    # Get calibration subset (matched + isolated + central)
    matched_calib_mask = calib_mask[matched_det_idx]
    
    if np.sum(matched_calib_mask) < phot_cfg['min_calibration_stars']:
        if verbose:
            logging.warning(f"    Too few isolated+central matches: {np.sum(matched_calib_mask)}")
        return False, None
    
    # =========================================================================
    # STEP 1: Prepare calibration star data
    # =========================================================================
    
    # Indices of calibration stars (matched + isolated + central)
    calib_det_indices = matched_det_idx[matched_calib_mask]
    calib_cat_indices = matched_cat_idx[matched_calib_mask]
    n_calib = len(calib_det_indices)
    
    # Get catalog magnitudes for calibration stars
    mc_calib_all = catalog_filtered[filter_name].values[calib_cat_indices].astype(np.float64)
    err_col = f'e{filter_name}'
    mce_raw = catalog_filtered[err_col].values[calib_cat_indices]
    mce_calib_all = np.zeros(len(mce_raw), dtype=np.float64)
    for i, val in enumerate(mce_raw):
        try:
            mce_calib_all[i] = float(val) if val is not None and val != '' else default_cat_err
        except (ValueError, TypeError):
            mce_calib_all[i] = default_cat_err
    mce_calib_all = np.where(np.isfinite(mce_calib_all) & (mce_calib_all > 0), 
                             mce_calib_all, default_cat_err)
    
    # =========================================================================
    # STEP 2: Fixed-aperture photometry
    # =========================================================================
    
    fixed_aperture = phot_cfg.get('aperture_radius', 2.5)
    target_rms = phot_cfg.get('target_rms', None)
    
    # Measure all sources at the fixed aperture
    f_opt, fe_opt = aperture_photometry_with_noise(positions, data_sub, photometry_noise, fixed_aperture)
    
    # Instrumental magnitudes
    final_mag_inst = np.full(len(sources), 99.0)
    final_mag_err = np.full(len(sources), 99.0)
    valid_all = f_opt > 0
    final_mag_inst[valid_all] = inst_zp - 2.5 * np.log10(f_opt[valid_all])
    final_mag_err[valid_all] = 2.5 / np.log(10) * fe_opt[valid_all] / f_opt[valid_all]
    
    # Fit ZP with calibration stars
    cmi_opt = final_mag_inst[calib_det_indices]
    cmie_opt = final_mag_err[calib_det_indices]
    valid_calib = (cmi_opt < 50) & (mc_calib_all < 50) & (cmie_opt < max_inst_err)
    
    if np.sum(valid_calib) < phot_cfg['min_calibration_stars']:
        if verbose:
            logging.warning(f"    Too few valid calibration stars at r={fixed_aperture:.1f} px")
        return False, None
    
    zp, zp_err, rms, final_mask, n_used, n_total = fit_zeropoint(
        cmi_opt[valid_calib], mc_calib_all[valid_calib],
        cmie_opt[valid_calib], mce_calib_all[valid_calib],
        target_rms=target_rms,
        min_stars=phot_cfg['min_calibration_stars'], verbose=False
    )
    
    mi_for_zp = cmi_opt[valid_calib]
    mei_for_zp = cmie_opt[valid_calib]
    mc_for_zp = mc_calib_all[valid_calib]
    mce_for_zp = mce_calib_all[valid_calib]
    
    if verbose:
        logging.info(f"    Aperture: r={fixed_aperture:.2f} px")
        logging.info(f"    ZP={zp:.3f}±{zp_err:.3f}, RMS={rms:.3f}, "
                     f"{n_used} stars, r={fixed_aperture:.1f}px")
    
    # =========================================================================
    # STEP 3: Apply zeropoint to get calibrated magnitudes
    # =========================================================================
    
    mag_cal = final_mag_inst + zp
    mag_cal_err = final_mag_err
    
    # =========================================================================
    # STEP 4: Generate calibration plot
    # =========================================================================
    
    plot_path = os.path.join(output_dir, filename.replace('.fits', '_photcal.png'))
    generate_photometry_plot(
        mi_for_zp, mc_for_zp, mei_for_zp, mce_for_zp, final_mask,
        zp, zp_err, rms, n_used, filter_name, filename, plot_path
    )
    
    if verbose:
        logging.info(f"    Saved calibration plot: {os.path.basename(plot_path)}")
    
    # =========================================================================
    # STEP 5: Build full source catalog with flags
    # =========================================================================
    
    # Convert pixel to world coordinates for all sources
    ra_all, dec_all = wcs.all_pix2world(positions[:, 0], positions[:, 1], 0)
    
    # Build catalog data with isolation/border flags
    sources_data = []
    for i in range(len(sources)):
        # Compute flag: 0=isolated+central, 1=crowded, 2=border, 3=crowded+border
        is_isolated = isolation_mask[i]
        is_central = central_mask[i]
        flag = 0
        if not is_isolated:
            flag += 1  # crowded
        if not is_central:
            flag += 2  # border
        
        src = {
            'ra': ra_all[i],
            'dec': dec_all[i],
            'x': positions[i, 0],
            'y': positions[i, 1],
            'mag_inst': final_mag_inst[i],
            'e_mag_inst': final_mag_err[i],
            'mag_cat': -99.0,
            'e_mag_cat': -99.0,
            'mag_cal': mag_cal[i],
            'e_mag_cal': mag_cal_err[i],
            'aperture': fixed_aperture,
            'flag': flag
        }
        
        # Check if this source was matched to catalog
        match_idx = np.where(matched_det_idx == i)[0]
        if len(match_idx) > 0:
            cat_idx = matched_cat_idx[match_idx[0]]
            try:
                src['mag_cat'] = float(catalog_filtered[filter_name].values[cat_idx])
            except (ValueError, TypeError):
                src['mag_cat'] = -99.0
            try:
                src['e_mag_cat'] = float(catalog_filtered[err_col].values[cat_idx])
                if not np.isfinite(src['e_mag_cat']) or src['e_mag_cat'] <= 0:
                    src['e_mag_cat'] = default_cat_err
            except (ValueError, TypeError):
                src['e_mag_cat'] = default_cat_err
        
        sources_data.append(src)
    
    # Compute limiting magnitude from image noise properties
    # Uses the noise map + ZP to estimate the faintest detectable source at S/N=3
    mag_lim = compute_limiting_magnitude(
        zp, photometry_noise, fixed_aperture,
        inst_zp=inst_zp, target_snr=3.0
    )
    
    if verbose:
        logging.info(f"    Limiting magnitude (S/N=3): {mag_lim:.2f} mag")
    
    # =========================================================================
    # STEP 5b: Forced photometry at target positions (-f flag)
    # =========================================================================
    # For each target from the -f file, if no detected source falls within the
    # error radius, perform aperture photometry at the exact target position.
    # This gives a measurement (or meaningful upper limit) even when the source
    # is too faint for blind detection.
    forced_phot_results = []
    if targets and wcs is not None:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        
        max_forced_err = phot_cfg.get('max_forced_mag_err', 0.33)
        
        for tgt in targets:
            try:
                tgt_x, tgt_y = wcs.all_world2pix(tgt['ra_deg'], tgt['dec_deg'], 0)
            except Exception:
                continue
            
            # Skip if outside image
            if tgt_x < 0 or tgt_x >= nx or tgt_y < 0 or tgt_y >= ny:
                continue
            
            # Check if there's already a detected source within the error radius
            already_detected = False
            tgt_coord = SkyCoord(tgt['ra_deg'], tgt['dec_deg'], unit=u.deg)
            for sd in sources_data:
                src_coord = SkyCoord(sd['ra'], sd['dec'], unit=u.deg)
                if tgt_coord.separation(src_coord).arcsec <= tgt['radius_arcsec']:
                    already_detected = True
                    break
            
            if already_detected:
                continue
            
            # Forced aperture photometry at the exact target position
            tgt_pos = np.array([[tgt_x, tgt_y]])
            f_forced, fe_forced = aperture_photometry_with_noise(
                tgt_pos, data_sub, photometry_noise, fixed_aperture
            )
            
            flux_val = f_forced[0]
            flux_err = fe_forced[0]
            
            # Compute instrumental and calibrated magnitudes
            if flux_val > 0 and np.isfinite(flux_val):
                mag_inst_forced = inst_zp - 2.5 * np.log10(flux_val)
                mag_err_forced = 2.5 / np.log(10) * flux_err / flux_val
                mag_cal_forced = mag_inst_forced + zp
            else:
                # Negative or zero flux — no meaningful measurement possible.
                # Skip this target (the epoch will appear as a regular
                # non-detection / upper limit in the notebook).
                if verbose:
                    logging.info(f"    Forced photometry at ({tgt['ra_deg']:.5f}, "
                                 f"{tgt['dec_deg']:.5f}): no flux (skipped)")
                continue
            
            forced_src = {
                'ra': tgt['ra_deg'],
                'dec': tgt['dec_deg'],
                'x': tgt_x,
                'y': tgt_y,
                'mag_inst': mag_inst_forced,
                'e_mag_inst': mag_err_forced,
                'mag_cat': -99.0,
                'e_mag_cat': -99.0,
                'mag_cal': mag_cal_forced,
                'e_mag_cal': mag_err_forced,
                'aperture': fixed_aperture,
                'flag': 4  # forced photometry
            }
            
            sources_data.append(forced_src)
            forced_phot_results.append(forced_src)
            
            if verbose:
                if mag_err_forced < max_forced_err and mag_cal_forced < 90:
                    logging.info(f"    Forced photometry at ({tgt['ra_deg']:.5f}, "
                                 f"{tgt['dec_deg']:.5f}): {mag_cal_forced:.3f} ± "
                                 f"{mag_err_forced:.3f} mag")
                else:
                    logging.info(f"    Forced photometry at ({tgt['ra_deg']:.5f}, "
                                 f"{tgt['dec_deg']:.5f}): non-detection "
                                 f"(err={mag_err_forced:.3f})")
        
        if verbose and forced_phot_results:
            n_good = sum(1 for fp in forced_phot_results 
                         if fp['e_mag_cal'] < max_forced_err and fp['mag_cal'] < 90)
            logging.info(f"    Forced photometry: {n_good}/{len(forced_phot_results)} "
                         f"targets with error < {max_forced_err} mag")
    
    # Generate detection map (all sources circled, with target overlay if -f given)
    if config.get('detection_map', {}).get('enabled', True):
        det_map_path = os.path.join(output_dir, filename.replace('.fits', '_detections.png'))
        pixel_scale = config['detector'].get('pixel_scale', 1.221)
        generate_detection_map(data_sub, sources, fixed_aperture, filename, det_map_path, config,
                               targets=targets, wcs=wcs, sources_data=sources_data,
                               mag_lim=mag_lim, pixel_scale=pixel_scale)
        if verbose:
            logging.info(f"    Saved detection map: {os.path.basename(det_map_path)}")
    
    # Save photometry catalog with all parameters
    catalog_txt_path = os.path.join(output_dir, filename.replace('.fits', '_photometry.txt'))
    save_photometry_catalog(
        catalog_txt_path, sources_data,
        zp, zp_err, rms, n_used, fixed_aperture, filter_name, filename,
        config,
        mag_lim=mag_lim, 
        threshold_sigma=phot_cfg['threshold_sigma'],
        min_pixels=phot_cfg['min_pixels'],
        n_total=n_total,
        date_obs=header.get('DATE-OBS'),
        exptime=header.get('EXPTIME'),
        object_name=header.get('OBJECT', None),
        is_coadd=is_coadd,
        ncoadd=ncoadd
    )
    
    if verbose:
        logging.info(f"    Saved photometry catalog: {os.path.basename(catalog_txt_path)}")
    
    # Calculate quality flags using helper functions
    rms_quality = calculate_rms_quality(rms, config)
    rejection_quality, _ = calculate_rejection_quality(n_used, n_total, config)
    
    # Return calibration info
    return True, {
        'zeropoint': zp,
        'zeropoint_err': zp_err,
        'rms': rms,
        'aperture_radius': fixed_aperture,
        'n_stars': n_used,
        'mag_lim': mag_lim,
        'filter': filter_name,
        'filename': filename,
        'exptime': exptime,
        'rms_quality': rms_quality,
        'rejection_quality': rejection_quality
    }


def astrometry_on_coadd(coadd_path, catalog_path, config, scale_constraint, verbose=False):
    """Perform astrometry on coadd file."""
    with fits.open(coadd_path) as hdul:
        data = hdul[0].data.astype(np.float32)
        header = hdul[0].header
    
    filter_name = header.get('FILTER', 'H').strip()
    
    # Load catalog
    catalog_df = pd.read_csv(catalog_path)
    
    # Try astrometry
    wcs, n_matches, attempt_info = try_astrometry(data, header, catalog_df, config, scale_constraint, filter_name, verbose)
    
    return wcs, n_matches, attempt_info

def generate_preview_jpg(fits_path, config):
    """Generate JPG preview of FITS file.
    
    Parameters:
    -----------
    fits_path : str
        Path to FITS file
    config : dict
        Configuration dictionary with preview settings
    """
    preview_cfg = config.get('preview', {})
    if not preview_cfg.get('enabled', False):
        return
    
    try:
        # Read FITS data
        with fits.open(fits_path) as hdul:
            data = hdul[0].data
            filename = os.path.basename(fits_path)
        
        # Skip if data is invalid
        if data is None or not np.isfinite(data).any():
            return
        
        # Get configuration
        use_central_stats = preview_cfg.get('use_central_stats', False)
        figsize = preview_cfg.get('figsize', [8, 8])
        dpi = preview_cfg.get('dpi', 100)
        cmap = preview_cfg.get('colormap', 'gray')
        invert = preview_cfg.get('invert', False)
        fontsize = preview_cfg.get('title_fontsize', 14)
        
        # Calculate display range
        if use_central_stats:
            # Use central region statistics
            central_fraction = preview_cfg.get('central_fraction', 0.8)
            vmin_sigma = preview_cfg.get('vmin_sigma', -0.5)
            vmax_sigma = preview_cfg.get('vmax_sigma', 4.0)
            
            # Extract central region
            central_data = get_central_region(data, central_fraction)
            
            # Calculate statistics from central region
            median = np.nanmedian(central_data)
            sigma = np.nanstd(central_data)
            
            # Set vmin/vmax based on sigma scaling
            vmin = median + vmin_sigma * sigma
            vmax = median + vmax_sigma * sigma
        else:
            # Use percentiles (legacy method)
            valid_data = data[np.isfinite(data)]
            if len(valid_data) == 0:
                return
            vmin = np.percentile(valid_data, preview_cfg.get('percentile_low', 1.0))
            vmax = np.percentile(valid_data, preview_cfg.get('percentile_high', 99.5))
        
        # Invert colormap if requested (for black stars on white background)
        if invert:
            cmap = cmap + '_r'
        
        # Create figure without axes
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Display image (only data matrix, no decorations)
        ax.imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        
        # Remove all axes, ticks, labels, and borders
        ax.set_axis_off()
        
        # Save as JPG with tight layout (no whitespace)
        jpg_path = fits_path.replace('.fits', '.jpg')
        plt.savefig(jpg_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close(fig)
        
        logging.debug(f"    Generated preview: {os.path.basename(jpg_path)}")
        
    except Exception as e:
        logging.warning(f"    Failed to generate preview for {os.path.basename(fits_path)}: {e}")


def save_with_astrometry(file_path, wcs, output_dir, config):
    """Save file with astrometry."""
    with fits.open(file_path) as hdul:
        header = hdul[0].header
        data = hdul[0].data
        
        # Update WCS
        wcs_header = wcs.to_header()
        for key in wcs_header:
            header[key] = wcs_header[key]
        
        header['ASTROP'] = 2
        header['DATE'] = format_date_like_dateobs(header.get('DATE-OBS', ''))
        
        basename = os.path.basename(file_path)
        output_filename = basename.replace('.fits', '_astro.fits')
        output_path = os.path.join(output_dir, output_filename)
        
        header['FILENAME'] = output_filename
        header['HISTORY'] = 'Astrometry calibrated'
        
        # Write with potentially interpolated data
        primary_hdu = fits.PrimaryHDU(data, header=header)
        hdu_list = [primary_hdu]
        
        # Preserve NOISE extension if present
        if len(hdul) > 1 and hdul[1].name == 'NOISE':
            hdu_list.append(hdul[1].copy())
        
        fits.HDUList(hdu_list).writeto(output_path, overwrite=True)
    
    # Generate preview JPG
    generate_preview_jpg(output_path, config)
    
    return output_path

def save_without_astrometry(file_path, output_dir, config):
    """Save file without astrometry."""
    with fits.open(file_path) as hdul:
        header = hdul[0].header
        data = hdul[0].data
        
        header['ASTROP'] = 1
        header['DATE'] = format_date_like_dateobs(header.get('DATE-OBS', ''))
        
        basename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, basename)
        
        header['FILENAME'] = basename
        header['HISTORY'] = 'Astrometry failed'
        
        # Write with potentially interpolated data
        primary_hdu = fits.PrimaryHDU(data, header=header)
        hdu_list = [primary_hdu]
        
        # Preserve NOISE extension if present
        if len(hdul) > 1 and hdul[1].name == 'NOISE':
            hdu_list.append(hdul[1].copy())
        
        fits.HDUList(hdu_list).writeto(output_path, overwrite=True)
    
    # Generate preview JPG
    generate_preview_jpg(output_path, config)
    
    return output_path


def check_standard_stars(photometry_results, reduced_dir, config, verbose=False):
    """Check standard stars (PROCTYPE=1) photometry results.
    
    For each standard star file with photometry:
    1. Find all sources within tolerance radius (arcsec) of RA/DEC
    2. Pick the brightest source (lowest mag_cal)
    3. Check if catalog magnitude is available
    4. Calculate difference between calibrated and catalog magnitude
    
    Returns:
    --------
    standard_checks : list of dicts
        Results for each standard star
    """
    phot_cfg = config['photometry']
    tolerance_arcsec = phot_cfg.get('standard_check_tolerance_arcsec', 10.0)
    
    standard_checks = []
    
    for result in photometry_results:
        filename = result['filename']
        fits_path = os.path.join(reduced_dir, filename)
        
        # Check PROCTYPE
        try:
            with fits.open(fits_path) as hdul:
                header = hdul[0].header
                proctype = header.get('PROCTYPE', 2)
                
                if proctype != 1:  # Not a standard star
                    continue
                
                # Get WCS and standard star position
                wcs = WCS(header)
                ra_std = header.get('RA', None)
                dec_std = header.get('DEC', None)
                
                if ra_std is None or dec_std is None:
                    if verbose:
                        logging.warning(f"  {filename}: No RA/DEC in header")
                    continue
                
                # Convert RA/DEC to pixel coordinates
                x_std, y_std = wcs.all_world2pix(ra_std, dec_std, 0)
                
                # Convert tolerance from arcsec to pixels
                pixel_scale = config['detector']['pixel_scale']
                tolerance_px = tolerance_arcsec / pixel_scale
                
        except Exception as e:
            if verbose:
                logging.warning(f"  {filename}: Error reading FITS: {e}")
            continue
        
        # Read photometry catalog
        phot_file = fits_path.replace('.fits', '_photometry.txt')
        
        if not os.path.exists(phot_file):
            if verbose:
                logging.warning(f"  {filename}: No photometry file")
            continue
        
        try:
            # Read photometry catalog (skip header lines starting with #)
            phot_data = []
            with open(phot_file, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 11:  # ra dec x y mag_inst e_mag_inst mag_cat e_mag_cat mag_cal e_mag_cal flag
                        phot_data.append({
                            'ra': float(parts[0]),
                            'dec': float(parts[1]),
                            'x': float(parts[2]),
                            'y': float(parts[3]),
                            'mag_inst': float(parts[4]),
                            'e_mag_inst': float(parts[5]),
                            'mag_cat': float(parts[6]),
                            'e_mag_cat': float(parts[7]),
                            'mag_cal': float(parts[8]),
                            'e_mag_cal': float(parts[9]),
                            'flag': int(parts[10])
                        })
            
            if len(phot_data) == 0:
                standard_checks.append({
                    'filename': filename,
                    'filter': result['filter'],
                    'status': 'No detections'
                })
                continue
            
            # Find all sources within tolerance and pick brightest
            sources_in_tolerance = []
            for source in phot_data:
                dist = np.sqrt((source['x'] - x_std)**2 + (source['y'] - y_std)**2)
                if dist <= tolerance_px:
                    source['distance_px'] = dist
                    sources_in_tolerance.append(source)
            
            # Check if any sources within tolerance
            if len(sources_in_tolerance) == 0:
                standard_checks.append({
                    'filename': filename,
                    'filter': result['filter'],
                    'status': 'No detections'
                })
                continue
            
            # Pick brightest source (lowest mag_cal)
            closest_source = min(sources_in_tolerance, key=lambda s: s['mag_cal'])
            min_dist = closest_source['distance_px']
            
            # Check if catalog magnitude is available
            if closest_source['mag_cat'] == -99.0 or abs(closest_source['mag_cat'] + 99.0) < 0.01:
                standard_checks.append({
                    'filename': filename,
                    'filter': result['filter'],
                    'status': 'No catalog',
                    'mag_cal': closest_source['mag_cal'],
                    'e_mag_cal': closest_source['e_mag_cal'],
                    'distance_px': min_dist
                })
                continue
            
            # Calculate difference
            mag_diff = closest_source['mag_cal'] - closest_source['mag_cat']
            mag_diff_err = np.sqrt(closest_source['e_mag_cal']**2 + closest_source['e_mag_cat']**2)
            
            # Calculate ZP_check quality
            zp_check = calculate_zp_check_quality(mag_diff, config, 'OK')
            
            standard_checks.append({
                'filename': filename,
                'filter': result['filter'],
                'status': 'OK',
                'mag_inst': closest_source['mag_inst'],
                'e_mag_inst': closest_source['e_mag_inst'],
                'mag_cal': closest_source['mag_cal'],
                'e_mag_cal': closest_source['e_mag_cal'],
                'mag_cat': closest_source['mag_cat'],
                'e_mag_cat': closest_source['e_mag_cat'],
                'mag_diff': mag_diff,
                'e_mag_diff': mag_diff_err,
                'distance_px': min_dist,
                'zp_check': zp_check
            })
            
        except Exception as e:
            if verbose:
                logging.warning(f"  {filename}: Error processing photometry: {e}")
            continue
    
    return standard_checks


def calculate_zp_check_quality(diff, config, status='OK'):
    """Calculate ZP_check quality based on difference.
    
    Thresholds from config['photometry']['quality_thresholds']['zp_comparison']:
    - VERY GOOD: |diff| < very_good
    - GOOD: |diff| < good
    - MEDIUM: |diff| < medium
    - POOR: |diff| >= medium
    - UNKNOWN: no comparison available
    """
    if status != 'OK':
        return 'UNKNOWN'
    
    # Get thresholds from config
    zp_thresh = config['photometry']['quality_thresholds']['zp_comparison']
    
    abs_diff = abs(diff)
    if abs_diff < zp_thresh['very_good']:
        return "VERY GOOD"
    elif abs_diff < zp_thresh['good']:
        return "GOOD"
    elif abs_diff < zp_thresh['medium']:
        return "MEDIUM"
    elif abs_diff < zp_thresh['poor']:
        return "POOR"
    else:
        return "VERY POOR"


def update_zp_check_in_photometry_files(photometry_results, standard_checks, zp_checks, reduced_dir, config, verbose=False):
    """Update photometry files with ZP_check quality keyword.
    
    For PROCTYPE=1 (standards): based on mag_diff from standard_checks
    For PROCTYPE=2 (non-standards): based on zp_diff from zp_checks
    
    Thresholds:
    - VERY GOOD: |diff| < 0.05
    - GOOD: |diff| < 0.1
    - MEDIUM: |diff| < 0.2
    - POOR: |diff| >= 0.2
    - UNKNOWN: no comparison available
    """
    # Create lookup dictionaries
    standard_check_map = {check['filename']: check for check in standard_checks}
    zp_check_map = {check['filename']: check for check in zp_checks}
    
    for result in photometry_results:
        filename = result['filename']
        phot_file = os.path.join(reduced_dir, filename.replace('.fits', '_photometry.txt'))
        
        if not os.path.exists(phot_file):
            continue
        
        # Determine PROCTYPE
        try:
            fits_path = os.path.join(reduced_dir, filename)
            with fits.open(fits_path) as hdul:
                proctype = hdul[0].header.get('PROCTYPE', 2)
        except:
            proctype = 2
        
        # Calculate ZP_check quality
        if proctype == 1:
            # Standard star - use mag_diff from standard_checks
            if filename in standard_check_map:
                check = standard_check_map[filename]
                zp_check = calculate_zp_check_quality(check.get('mag_diff', 0), config, check['status'])
            else:
                zp_check = "UNKNOWN"
        else:
            # Non-standard - use zp_diff from zp_checks
            if filename in zp_check_map:
                check = zp_check_map[filename]
                zp_check = calculate_zp_check_quality(check.get('zp_diff', 0), config, check['status'])
            else:
                zp_check = "UNKNOWN"
        
        # Update photometry file
        try:
            # Read current content
            with open(phot_file, 'r') as f:
                lines = f.readlines()
            
            # Find insertion point (after RMS quality line)
            insert_idx = None
            for i, line in enumerate(lines):
                if line.startswith('# RMS quality:'):
                    insert_idx = i + 1
                    break
            
            if insert_idx is not None:
                # Insert ZP_check line
                lines.insert(insert_idx, f"# ZP_check: {zp_check}\n")
                
                # Write back
                with open(phot_file, 'w') as f:
                    f.writelines(lines)
                
                if verbose:
                    logging.debug(f"    Updated {os.path.basename(phot_file)} with ZP_check: {zp_check}")
        
        except Exception as e:
            if verbose:
                logging.warning(f"    Failed to update {os.path.basename(phot_file)}: {e}")


def check_zeropoint_consistency(photometry_results, standard_checks, reduced_dir, config, verbose=False):
    """Check zeropoint consistency between catalog-based and standard-based calibration.
    
    For each non-standard calibrated file:
    1. Find closest standard star coadd in time (same filter)
    2. From standard check results: use the star identified within tolerance of standard's RA/DEC
    3. Compute ZP from that star: ZP_std = mag_cat - mag_inst
    4. Correct for exptime difference: ZP_standard = ZP_std + 2.5 × log10(exptime_sci / exptime_std)
    5. Compare with catalog-based ZP
    
    Parameters:
    -----------
    standard_checks : list of dicts
        Results from check_standard_stars() containing mag_cat, mag_cal, etc.
    
    Returns:
    --------
    zp_checks : list of dicts
        Results for each non-standard file with both ZP estimates
    """
    zp_checks = []
    
    # Separate standard and non-standard files
    standard_files = []
    non_standard_files = []
    
    for result in photometry_results:
        filename = result['filename']
        fits_path = os.path.join(reduced_dir, filename)
        
        try:
            with fits.open(fits_path) as hdul:
                header = hdul[0].header
                proctype = header.get('PROCTYPE', 2)
                date_obs = header.get('DATE-OBS', None)
                
                if date_obs is None:
                    continue
                
                result_with_time = result.copy()
                result_with_time['date_obs'] = date_obs
                result_with_time['time'] = Time(date_obs, format='isot')
                result_with_time['proctype'] = proctype
                
                if proctype == 1:  # Standard
                    standard_files.append(result_with_time)
                else:  # Non-standard (science)
                    non_standard_files.append(result_with_time)
        except Exception as e:
            if verbose:
                logging.warning(f"  Error reading {filename}: {e}")
            continue
    
    if not standard_files:
        if verbose:
            logging.info("  No standard files available for comparison")
        return zp_checks
    
    if not non_standard_files:
        if verbose:
            logging.info("  No non-standard files to check")
        return zp_checks
    
    # Create lookup map for standard check results
    standard_check_map = {check['filename']: check for check in standard_checks}
    
    # For each non-standard file, find closest standard in same filter
    for sci_file in non_standard_files:
        sci_filter = sci_file['filter']
        sci_time = sci_file['time']
        sci_filename = sci_file['filename']
        sci_exptime = sci_file['exptime']
        
        # Find standards in same filter
        same_filter_standards = [s for s in standard_files if s['filter'] == sci_filter]
        
        if not same_filter_standards:
            zp_checks.append({
                'filename': sci_filename,
                'filter': sci_filter,
                'zp_catalog': sci_file['zeropoint'],
                'zp_standard': None,
                'status': 'No standard',
                'zp_check': 'UNKNOWN'
            })
            continue
        
        # Find closest in time
        min_dt = float('inf')
        closest_std = None
        for std in same_filter_standards:
            dt = abs((sci_time - std['time']).sec)
            if dt < min_dt:
                min_dt = dt
                closest_std = std
        
        # Look up standard check results for this standard file
        std_filename = closest_std['filename']
        if std_filename not in standard_check_map:
            zp_checks.append({
                'filename': sci_filename,
                'filter': sci_filter,
                'zp_catalog': sci_file['zeropoint'],
                'zp_standard': None,
                'status': 'No std check',
                'zp_check': 'UNKNOWN'
            })
            continue
        
        std_check = standard_check_map[std_filename]
        
        # For science images, we use the zeropoint from the standard file
        # We don't need the standard star detection to be successful - just a valid ZP
        # Get ZP from standard file photometry result
        zp_std = closest_std.get('zeropoint', None)
        
        if zp_std is None or not np.isfinite(zp_std):
            zp_checks.append({
                'filename': sci_filename,
                'filter': sci_filter,
                'zp_catalog': sci_file['zeropoint'],
                'zp_standard': None,
                'status': 'No std ZP',
                'zp_check': 'UNKNOWN'
            })
            continue
        
        # Correct for exptime difference
        # ZP_standard = ZP_std + 2.5 × log10(exptime_sci / exptime_std)
        std_exptime = closest_std['exptime']
        if std_exptime > 0 and sci_exptime > 0:
            exptime_correction = 2.5 * np.log10(sci_exptime / std_exptime)
            zp_standard = zp_std + exptime_correction
        else:
            zp_standard = zp_std
            exptime_correction = 0.0
        
        # Calculate ZP_check quality
        zp_diff = sci_file['zeropoint'] - zp_standard
        zp_check = calculate_zp_check_quality(zp_diff, config, 'OK')
        
        zp_checks.append({
            'filename': sci_filename,
            'filter': sci_filter,
            'zp_catalog': sci_file['zeropoint'],
            'zp_standard': zp_standard,
            'zp_diff': zp_diff,
            'closest_standard': std_filename,
            'time_diff_sec': min_dt,
            'exptime_sci': sci_exptime,
            'exptime_std': std_exptime,
            'exptime_correction': exptime_correction,
            'status': 'OK',
            'zp_check': zp_check
        })
    
    return zp_checks


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    # Start timer
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
    # Load config
    config = load_config(args.config)
    
    # Setup directories
    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output if args.output else args.input)
    
    tmp_dir = os.path.join(output_dir, 'tmp')
    reduced_dir = os.path.join(output_dir, 'reduced')
    catalog_dir = os.path.join(output_dir, 'catalogs')
    
    # Clean output directories if requested
    if args.clean_output:
        for dir_to_clean in [tmp_dir, reduced_dir, catalog_dir]:
            if os.path.exists(dir_to_clean):
                shutil.rmtree(dir_to_clean)
                if args.verbose:
                    logging.info(f"Cleaned existing directory: {dir_to_clean}")
    
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(reduced_dir, exist_ok=True)
    os.makedirs(catalog_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(reduced_dir, 'pipelog.txt')
    setup_logging(log_file, args.verbose)
    
    logging.info("=" * 80)
    logging.info("REMIR Pipeline")
    logging.info("=" * 80)
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Config file: {args.config}")
    if args.target:
        logging.info(f"Target filter: astrometry/photometry only for OBJECT in {args.target}")
    
    # Parse target positions file (if provided via -f)
    user_targets = None
    if args.target_file:
        try:
            user_targets = parse_target_file(args.target_file)
            logging.info(f"Loaded {len(user_targets)} target positions from {args.target_file}")
        except Exception as e:
            logging.warning(f"Failed to parse target file {args.target_file}: {e}")
    
    logging.info("")
    
    # Step 1: Gunzip files
    gunzip_files(input_dir, args.verbose)
    
    # Step 2: Filter and prepare files
    filter_and_prepare_files(input_dir, config, args.verbose)
    
    # Step 3: Classify files
    classify_files(input_dir, tmp_dir, reduced_dir, config, args.verbose)
    
    # Step 4: Group and process old/new
    all_coadd_files = []
    coadd_to_aligned = {}  # Map coadd path to list of aligned file paths
    photometry_results = []  # Collect photometry results for summary

    for system in ['old', 'new']:
        system_dir = os.path.join(tmp_dir, system)
        
        if not os.path.exists(system_dir):
            continue
        
        logging.info("")
        logging.info(f"Processing {system} system...")
        
        # Group files
        groups = group_files(system_dir, config, args.verbose)
        
        # Validate groups
        valid_groups, incomplete_groups, defective_groups = validate_groups(groups, args.verbose)
        
        # Process groups
        processable_groups = valid_groups + incomplete_groups
        
        for group in processable_groups:
            is_incomplete = group in incomplete_groups
            
            # Sky subtraction (level matching + single sky + flat fielding)
            sky_path, skysub_files, group_name = process_group_sky_subtraction(
                group, config, system_dir, args.verbose
            )
            
            # Copy single sky file to reduced
            sky_reduced_path = os.path.join(reduced_dir, os.path.basename(sky_path))
            shutil.copy2(sky_path, sky_reduced_path)
            generate_preview_jpg(sky_reduced_path, config)
            
            # Alignment (using drizzling algorithm)
            aligned_files = align_frames(skysub_files, config, system, system_dir, args.verbose)
            
            # Coadd
            coadd_path = coadd_aligned_frames(
                aligned_files, group_name, system_dir, is_incomplete, config, args.verbose
            )
            
            all_coadd_files.append(coadd_path)
            # Store mapping from coadd to its aligned files
            coadd_to_aligned[coadd_path] = [f['path'] for f in aligned_files]
    
    # Step 5: Group coadds by position and download catalogs
    logging.info("")
    logging.info("Grouping coadds by position and downloading catalogs...")
    
    # When -t/--target is given, only download catalogs for matching OBJECTs
    if args.target:
        coadds_for_catalogs = []
        for cp in all_coadd_files:
            with fits.open(cp) as _hdul:
                obj = _hdul[0].header.get('OBJECT', '').strip()
            if obj in args.target:
                coadds_for_catalogs.append(cp)
        logging.info(f"  Target filter active: {len(coadds_for_catalogs)}/{len(all_coadd_files)} coadds match {args.target}")
    else:
        coadds_for_catalogs = all_coadd_files
    
    position_groups = group_coadds_by_position(coadds_for_catalogs, config, args.verbose)
    catalog_map = download_catalogs_for_groups(position_groups, config, catalog_dir, args.verbose)
    
    # Check if any catalogs were downloaded
    if not catalog_map:
        logging.error("")
        logging.error("=" * 80)
        logging.error("FATAL ERROR: Failed to download any catalogs")
        logging.error("=" * 80)
        logging.error("Possible causes:")
        logging.error("  1. No internet connection")
        logging.error("  2. INAF catalog service is down")
        logging.error("  3. All catalog downloads failed (check errors above)")
        logging.error("")
        logging.error("Cannot proceed with astrometry and photometry without catalogs.")
        logging.error("Please check your internet connection and try again.")
        logging.error("=" * 80)
        sys.exit(1)
    
    logging.info(f"Successfully downloaded catalogs for {len(catalog_map)} coadd groups")
    
    # Step 6: Astrometry
    logging.info("")
    logging.info("Performing astrometry...")
    
    # Track success signatures (thresh-npix-filter-reflection) for global summary
    success_signatures = defaultdict(int)

    for coadd_path in all_coadd_files:
        if coadd_path not in catalog_map:
            save_without_astrometry(coadd_path, reduced_dir, config)
            for aligned_path in coadd_to_aligned.get(coadd_path, []):
                save_without_astrometry(aligned_path, reduced_dir, config)
            continue

        catalog_path = catalog_map[coadd_path]

        # Major division for each coadd (always printed)
        log_big_divider(f"Processing {os.path.basename(coadd_path)}")

        # Read header once for filter and object checks
        with fits.open(coadd_path) as _hdul:
            _filter_name = _hdul[0].header.get('FILTER', '').strip()
            _object_name = _hdul[0].header.get('OBJECT', '').strip()

        # If -t/--target was given, skip astrometry/photometry for non-matching OBJECTs
        if args.target and _object_name not in args.target:
            logging.info(f"  OBJECT '{_object_name}' not in target list — skipping astrometry/photometry")
            save_without_astrometry(coadd_path, reduced_dir, config)
            for aligned_path in coadd_to_aligned.get(coadd_path, []):
                save_without_astrometry(aligned_path, reduced_dir, config)
            continue

        # Check if filter should skip astrometry (e.g., GRISM, dispersed modes)
        skip_filters = config.get('astrometry', {}).get('skip_filters', [])
        if _filter_name in skip_filters:
            logging.info(f"  Filter '{_filter_name}' in skip_filters — copying as-is")
            save_without_astrometry(coadd_path, reduced_dir, config)
            for aligned_path in coadd_to_aligned.get(coadd_path, []):
                save_without_astrometry(aligned_path, reduced_dir, config)
            continue

        # Try astrometry
        wcs, n_matches, attempt_info = astrometry_on_coadd(
            coadd_path, catalog_path, config, args.scale_constraint, args.verbose
        )

        # Get aligned files belonging to this coadd from the mapping
        aligned_file_paths = coadd_to_aligned.get(coadd_path, [])

        if args.verbose:
            logging.info(f"    Found {len(aligned_file_paths)} aligned files for this group")

        if wcs is not None:
            if args.verbose:
                logging.info(f"    Success! {n_matches} matches")

            # Save coadd with astrometry
            astro_coadd_path = save_with_astrometry(coadd_path, wcs, reduced_dir, config)

            # Save aligned files with astrometry and collect their output paths
            astro_aligned_paths = []
            for aligned_path in aligned_file_paths:
                astro_aligned_path = save_with_astrometry(aligned_path, wcs, reduced_dir, config)
                astro_aligned_paths.append(astro_aligned_path)
                STATS['astrometrized'] += 1  # Count each aligned file too

            # Update stats
            STATS['astrometrized'] += 1
            # Count signature (thresh-npix-filter-reflection)
            if attempt_info is not None:
                sig = f"th={attempt_info['thresh']}-np={attempt_info['npix']}-f={attempt_info['filter']}-ref={int(attempt_info['is_reflection'])}"
                success_signatures[sig] += 1

            # Photometric calibration on the astrometrized coadd (only for JHK filters)
            try:
                phot_success, calib_info = photometric_calibration(
                    astro_coadd_path, catalog_path, config, reduced_dir, args.verbose,
                    targets=user_targets
                )
                if phot_success and calib_info is not None:
                    STATS['photometrized'] += 1
                    photometry_results.append(calib_info)
                    if args.verbose:
                        logging.info(f"    Coadd photometry: ZP = {calib_info['zeropoint']:.3f}")
            except Exception as e:
                if args.verbose:
                    logging.warning(f"    Coadd photometry failed: {e}")
            
            # Independent photometric calibration on each aligned file
            if args.verbose:
                logging.info(f"    Running independent photometry on {len(astro_aligned_paths)} aligned files...")
            for astro_aligned_path in astro_aligned_paths:
                try:
                    aligned_phot_success, aligned_calib_info = photometric_calibration(
                        astro_aligned_path, catalog_path, config, reduced_dir, args.verbose,
                        targets=user_targets
                    )
                    if aligned_phot_success and aligned_calib_info is not None:
                        STATS['photometrized'] += 1
                        photometry_results.append(aligned_calib_info)
                except Exception as e:
                    if args.verbose:
                        logging.warning(f"      Photometry failed for {os.path.basename(astro_aligned_path)}: {e}")

        else:
            if args.verbose:
                logging.info("    Failed")

            # Save without astrometry
            save_without_astrometry(coadd_path, reduced_dir, config)

            # Save aligned files without astrometry
            for aligned_path in aligned_file_paths:
                save_without_astrometry(aligned_path, reduced_dir, config)

            STATS['failed_astrometry'] += 1
    
    # Cleanup
    if args.delete_tmp:
        logging.info("")
        logging.info("Deleting tmp directory...")
        shutil.rmtree(tmp_dir)
    
    logging.info("")
    # Final global summary
    logging.info("" )
    logging.info("=" * 80)
    logging.info("Pipeline completed! Global summary:")
    logging.info("=" * 80)
    total_coadds = len(all_coadd_files)
    astrom = STATS.get('astrometrized', 0)
    failed_ast = STATS.get('failed_astrometry', 0)
    n_skies = STATS.get('sky_frames', 0)
    n_coadds = STATS.get('coadds', 0)
    n_deleted = STATS.get('files_deleted', 0)
    n_photom = STATS.get('photometrized', 0)

    logging.info(f"  Total coadds processed: {total_coadds}")
    logging.info(f"  Coadds created: {n_coadds}")
    logging.info(f"  Sky frames created: {n_skies}")
    logging.info(f"  Files deleted during prep: {n_deleted}")
    
    # Calculate percentages
    total_ast_attempts = astrom + failed_ast
    astrom_pct = (astrom / total_ast_attempts * 100) if total_ast_attempts > 0 else 0
    failed_pct = (failed_ast / total_ast_attempts * 100) if total_ast_attempts > 0 else 0
    photom_pct = (n_photom / astrom * 100) if astrom > 0 else 0
    
    logging.info(f"  Astrometrized (coadds+aligned): {astrom} ({astrom_pct:.1f}%)  |  Failed astrometry: {failed_ast} ({failed_pct:.1f}%)")
    logging.info(f"  Photometrized: {n_photom}/{astrom} ({photom_pct:.1f}%) (JHK only)")

    # Print per-attempt signature breakdown if available
    try:
        sig_items = sorted(success_signatures.items(), key=lambda x: x[1], reverse=True)
        if sig_items:
            logging.info("")
            logging.info("Breakdown of successful attempts (thresh-npix-filter-reflection):")
            for sig, cnt in sig_items:
                logging.info(f"  {cnt:3d} x {sig}")
    except NameError:
        # success_signatures may not exist if no coadds processed
        pass
    
    # Print photometry breakdown
    if photometry_results:
        logging.info("")
        logging.info("Photometry breakdown:")
        logging.info(f"  {'Filename':<45} {'Filter':>6} {'ExpTime':>7} {'ZP':>7} {'eZP':>6} {'RMS':>6} {'Nstar':>5} {'MagLim':>7}")
        logging.info(f"  {'-'*45} {'-'*6} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*5} {'-'*7}")
        for r in photometry_results:
            logging.info(f"  {r['filename']:<45} {r['filter']:>6} {r['exptime']:>7.1f} {r['zeropoint']:>7.3f} {r['zeropoint_err']:>6.3f} {r['rms']:>6.3f} {r['n_stars']:>5} {r['mag_lim']:>7.2f}")
    
    # Check standard stars
    if photometry_results:
        logging.info("")
        logging.info("Standard star check:")
        standard_checks = check_standard_stars(photometry_results, reduced_dir, config, args.verbose)
        
        if standard_checks:
            logging.info(f"  {'Filename':<45} {'Filter':>6} {'MagCal':>7} {'eMagCal':>8} {'MagCat':>7} {'eMagCat':>8} {'Diff':>7} {'eDiff':>7} {'ZP_check':>12}")
            logging.info(f"  {'-'*45} {'-'*6} {'-'*7} {'-'*8} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*12}")
            for check in standard_checks:
                if check['status'] == 'No detections':
                    logging.info(f"  {check['filename']:<45} {check['filter']:>6} {'No detections':>67} {'UNKNOWN':>12}")
                elif check['status'] == 'No catalog':
                    logging.info(f"  {check['filename']:<45} {check['filter']:>6} {check['mag_cal']:>7.3f} {check['e_mag_cal']:>8.3f} {'No catalog':>23} {'UNKNOWN':>12}")
                else:  # OK
                    logging.info(f"  {check['filename']:<45} {check['filter']:>6} {check['mag_cal']:>7.3f} {check['e_mag_cal']:>8.3f} {check['mag_cat']:>7.3f} {check['e_mag_cat']:>8.3f} {check['mag_diff']:>7.3f} {check['e_mag_diff']:>7.3f} {check['zp_check']:>12}")
        else:
            logging.info("  No standard stars found")
    
    # Check zeropoint consistency (catalog vs standard-based)
    if photometry_results:
        logging.info("")
        logging.info("Zeropoint consistency check (catalog vs standard):")
        zp_checks = check_zeropoint_consistency(photometry_results, standard_checks, reduced_dir, config, args.verbose)
        
        if zp_checks:
            logging.info(f"  {'Filename':<45} {'Filter':>6} {'ZP_Cat':>7} {'ZP_Std':>7} {'Diff':>7} {'ZP_check':>12}")
            logging.info(f"  {'-'*45} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*12}")
            for check in zp_checks:
                if check['status'] != 'OK':
                    logging.info(f"  {check['filename']:<45} {check['filter']:>6} {check['zp_catalog']:>7.3f} {check['status']:>15} {'UNKNOWN':>12}")
                else:
                    logging.info(f"  {check['filename']:<45} {check['filter']:>6} {check['zp_catalog']:>7.3f} {check['zp_standard']:>7.3f} {check['zp_diff']:>7.3f} {check['zp_check']:>12}")
        else:
            logging.info("  No files to check")
    
    # Update photometry files with ZP_check keyword
    if photometry_results:
        logging.info("")
        logging.info("Updating photometry files with ZP_check keyword...")
        update_zp_check_in_photometry_files(
            photometry_results, standard_checks if 'standard_checks' in locals() else [], 
            zp_checks if 'zp_checks' in locals() else [], 
            reduced_dir, config, args.verbose
        )
    
    # Print quality flags summary
    if photometry_results:
        logging.info("")
        logging.info("Quality flags summary:")
        logging.info("")
        
        # RMS quality statistics
        rms_counts = {'VERY GOOD': 0, 'GOOD': 0, 'MEDIUM': 0, 'POOR': 0, 'VERY POOR': 0, 'UNKNOWN': 0}
        for r in photometry_results:
            rms_quality = r.get('rms_quality', 'UNKNOWN')
            if rms_quality in rms_counts:
                rms_counts[rms_quality] += 1
        
        total_files = len(photometry_results)
        logging.info("  RMS quality:")
        for quality in ['VERY GOOD', 'GOOD', 'MEDIUM', 'POOR', 'VERY POOR', 'UNKNOWN']:
            count = rms_counts[quality]
            percent = (count / total_files * 100) if total_files > 0 else 0
            logging.info(f"    {quality:12s}: {count:3d} files ({percent:5.1f}%)")
        
        # Rejection quality statistics
        rejection_counts = {'GOOD': 0, 'MEDIUM': 0, 'POOR': 0, 'UNKNOWN': 0}
        for r in photometry_results:
            rejection_quality = r.get('rejection_quality', 'UNKNOWN')
            if rejection_quality in rejection_counts:
                rejection_counts[rejection_quality] += 1
        
        logging.info("")
        logging.info("  Rejection quality:")
        for quality in ['GOOD', 'MEDIUM', 'POOR', 'UNKNOWN']:
            count = rejection_counts[quality]
            percent = (count / total_files * 100) if total_files > 0 else 0
            logging.info(f"    {quality:12s}: {count:3d} files ({percent:5.1f}%)")
        
        # ZP_check quality statistics (from standard_checks and zp_checks)
        if 'standard_checks' in locals() and 'zp_checks' in locals():
            zp_check_counts = {'VERY GOOD': 0, 'GOOD': 0, 'MEDIUM': 0, 'POOR': 0, 'UNKNOWN': 0}
            
            # Count from standard checks
            for check in standard_checks:
                zp_check_quality = check.get('zp_check', 'UNKNOWN')
                if zp_check_quality in zp_check_counts:
                    zp_check_counts[zp_check_quality] += 1
            
            # Count from zeropoint checks
            for check in zp_checks:
                zp_check_quality = check.get('zp_check', 'UNKNOWN')
                if zp_check_quality in zp_check_counts:
                    zp_check_counts[zp_check_quality] += 1
            
            total_zp_checks = sum(zp_check_counts.values())
            if total_zp_checks > 0:
                logging.info("")
                logging.info("  ZP_check quality:")
                for quality in ['VERY GOOD', 'GOOD', 'MEDIUM', 'POOR', 'UNKNOWN']:
                    count = zp_check_counts[quality]
                    percent = (count / total_zp_checks * 100) if total_zp_checks > 0 else 0
                    logging.info(f"    {quality:12s}: {count:3d} files ({percent:5.1f}%)")
    
    # Print elapsed time
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    
    logging.info("="  * 80)
    if hours > 0:
        logging.info(f"Total elapsed time: {hours}h {minutes}m {seconds:.1f}s")
    elif minutes > 0:
        logging.info(f"Total elapsed time: {minutes}m {seconds:.1f}s")
    else:
        logging.info(f"Total elapsed time: {seconds:.1f}s")
    logging.info("=" * 80)

if __name__ == "__main__":
    main()
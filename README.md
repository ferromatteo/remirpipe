# REMIR Pipeline

Automated reduction and calibration pipeline for REMIR (REM InfraRed) near-infrared imaging data from the REM telescope at La Silla Observatory.

## Overview

End-to-end processing of dithered NIR observations (J, H, K bands) from raw FITS files through sky subtraction, alignment, co-addition, astrometric calibration, and photometric calibration.

**Key features:**
- Median sky subtraction with post-median residual cleanup for clean skies in crowded fields
- Optional LOO + Iterative Gaussian mode for thermal wedge-prism arc removal (K band)
- Flux-preserving drizzle alignment with optional cross-match refinement
- Inverse-variance weighted co-addition with sigma clipping
- Quad-matching astrometry against 2MASS catalog
- Iterative source detection with deblending and residual re-extraction for crowded fields
- Automated photometric calibration with quality assessment and limiting magnitude
- Forced aperture photometry at user-specified target positions (`-f` flag)
- Target position overlay on detection maps (`-f` flag)

## Requirements

### Python Dependencies

```bash
pip install numpy scipy astropy photutils matplotlib pyyaml pandas sep requests
```

| Package | Min Version | Purpose |
|---------|-------------|---------|
| `numpy` | 1.20 | Array operations, linear algebra |
| `scipy` | 1.7 | cKDTree matching, Gaussian filtering |
| `astropy` | 5.0 | FITS I/O, WCS, coordinates, time |
| `photutils` | 1.5 | Aperture photometry |
| `sep` | 1.2 | Source Extraction (SExtractor backend) |
| `matplotlib` | 3.5 | Diagnostic plots, detection maps |
| `pyyaml` | 6.0 | Configuration parsing |
| `pandas` | 1.3 | Catalog handling |
| `requests` | — | HTTP catalog downloads |

### Calibration Files

Place in the `data_folder` directory (configured in `config.yaml`, default: `data_2026_01/`):

```
data_2026_01/
├── pixel_mask.fits              # Bad pixel mask (0=bad, 1=good)
├── J_dither0_flat.fits          # Master flat fields
├── J_dither72_flat.fits         # (per filter × dither angle)
├── J_dither144_flat.fits
├── H_dither0_flat.fits
└── ...
```

- **Pixel mask**: `pixel_mask.fits` — binary mask, 0=bad, 1=good
- **Flats**: `{FILTER}_dither{ANGLE}_flat.fits` for each filter (J/H/K) × dither angle (0/72/144/216/288)

A companion notebook (`cal_flat_and_bad_pix.ipynb`) can produce calibration files from raw flat observations.

## Installation

```bash
git clone https://github.com/ferromatteo/remirpipe.git
cd remirpipe
pip install -r requirements.txt
```

Then prepare calibration files and edit `config.yaml` for your setup.

## Usage

### Basic Command

```bash
python remirpipe.py -i /path/to/raw/data -v
```

### Command-Line Options

| Flag | Description | Default |
|------|-------------|---------|
| `-i`, `--input` | Input directory with raw FITS files | **required** |
| `-o`, `--output` | Output directory | same as input |
| `-c`, `--config` | Configuration YAML file | `config.yaml` (script dir) |
| `-v`, `--verbose` | Verbose output to console and log | off |
| `-d`, `--delete-tmp` | Delete `tmp/` directory after completion | off |
| `-s`, `--scale-constraint` | Enforce scale ∈ [0.95, 1.05] for astrometry | off |
| `-co`, `--clean-output` | Clean existing output directories before starting | off |
| `-t`, `--target` | Target OBJECT name(s) — astrometry/photometry only on these | all |
| `-f`, `--target-file` | Text file with target positions (RA DEC radius_arcsec) | none |

### Examples

```bash
# Standard reduction with verbose logging
python remirpipe.py -i ./raw_data/2026-01-15 -v

# Custom config, clean start, separate output
python remirpipe.py -i ./data -o ./reduced -c custom.yaml -co -v

# Strict astrometry with cleanup
python remirpipe.py -i ./data -s -d -v

# Process all data, astrometry/photometry only on specific targets
python remirpipe.py -i ./data -t NGC1234 M31 -v

# Target overlay on detection maps
python remirpipe.py -i ./data -f targets.txt -v
```

### Target Position File Format (`-f`)

Plain text file with one target per line: `RA DEC radius_arcsec`

Supports both sexagesimal and decimal degree formats:
```
13:58:09.72 -64:44:05.26 1.5
209.540500  -64.734794   2.0
```

When provided, the detection map PNG for each image marks each target position with a cyan cross and error circle. If a source is detected within the error radius, it is highlighted with its calibrated magnitude; otherwise the limiting magnitude is shown.

## Pipeline Workflow

### 1. File Preparation

- Gunzip `.fits.gz` files in-place
- Remove files with DITHID=98 or 99 (products from previous runs)
- Fix invalid header values (e.g., NaN in WINDDIR)
- Add keywords: FILENAME, PROCTYPE (0=FLAT, 1=STD, 2=SCI, -1=FOCUS)
- Apply bad pixel mask (sets masked pixels to NaN)

### 2. File Classification

- Classify by dither system: **old** (pre-2025, DWANGLE) vs **new** (post-2025, DITHANGL)
- FLAT/FOCUS → directly to `reduced/`
- SCI/STD → to `tmp/old/` or `tmp/new/` for processing

### 3. Grouping & Validation

- Group by: OBJECT + FILTER + OBSID + SUBID + time gap < 9 hours
- **Complete**: N = NDITHERS (typically 5) — full processing
- **Incomplete**: 3 ≤ N < NDITHERS — processable with flag
- **Defective**: N < 3 or N > NDITHERS — skipped

### 4. Sky Subtraction

Per group of N dithered frames:

1. **Level normalization**: Scale each frame so its central-region σ-clipped median equals the group mean median
2. **Median sky**: Pixel-wise median of all N leveled frames — no per-frame source masking (the median naturally rejects dithered stars)
3. **Post-median residual cleanup**: Detect surviving positive residuals in the sky (from bright stars that weren't fully rejected by the median), replace contaminated pixels by random draws from a clean annular neighborhood (5–15 px), preserving local noise statistics
4. **Sky subtraction + flat fielding**: `data_final = (data_leveled − sky) / flat`

**Processing formula**: `data_final = (data_raw × level_factor − sky_all) / flat`

#### LOO + Iterative Gaussian Mode (optional, per-filter)

When `sky_subtraction.loo_iterative_gauss.enabled: true` (per-filter override via `filter_enable`):

1. **Leave-one-out sky**: For each frame *i*, sky = median of the other N−1 frames (avoids self-subtraction of the thermal arc)
2. **Flat fielding** as normal
3. **Iterative Gaussian background removal** (3 iterations by default):
   - Source mask built once (SEP, threshold = `source_sigma` × σ, ellipse growth = `source_grow_base` px)
   - Each iteration *k*: σ_gauss = `base_sigma × decay^k` (default: 25.0 × 0.8^k → 25.0, 20.0, 16.0 px)
   - Background estimated via normalized convolution (handles masked pixels) and subtracted
   - Progressively removes large-scale to fine-scale residual structure

**Noise propagation**: LOO sky noise, flat division noise, and IterGauss smoothing variance are all propagated analytically (~1% total noise increase for default parameters).

### 5. Alignment

- **Geometric shifts**: Calculated from dither wedge angle + calibrated parameters
- **Optional refinement** (enabled by default): Cross-match sources between frames via displacement voting, then fit via:
  - **`similarity`** (default): Full similarity transform (rotation + scale + translation, 4 parameters)
  - **`shift`**: Pure translation (2 parameters) — use when inter-frame rotation is negligible
  - Iterative sigma-clipped matching with rescue mechanism (widens tolerance on failure)
  - Capped to brightest 50 sources
  - Configurable fallback to blind geometry
- **Drizzle**: Single-pass flux-preserving resampling based on geometric pixel overlap (pixfrac=0.8 default)

### 6. Co-addition

- **Inverse-variance weighted mean**: `coadd = Σ(data_i / σ_i²) / Σ(1 / σ_i²)`
- **Sigma clipping**: Reject pixels > 4σ from weighted mean (2 iterations, requires ≥3 frames)
- **Output**: FITS with NOISE and WEIGHT extensions

### 7. Catalog Download

- **2MASS PSC** via INAF service (cone search, 10′ default radius, with retries)
- **VSX** (Variable Star Index) cross-matched with 2MASS (0.5″ tolerance)
- Cached per position; images within 1′ share the same catalog
- With `-t`: catalogs only downloaded for matching OBJECTs

### 8. Astrometric Calibration

**Quad-matching algorithm:**

1. **Source detection** — Iterative SEP background refinement (3 iterations with source masking), deblending (`deblend_nthresh=64`, `deblend_cont=0.005`), residual re-extraction for faint sources near bright neighbors. Tries multiple parameter combinations: min_pixels=[10,5], threshold=[2.0,1.2].
2. **Geometric quad generation** — ~2500 quads from brightest 25 detected + 25 catalog sources
3. **Quad matching** — Euclidean distance on 4D hash codes (threshold: 0.05)
4. **Transform consensus** — Group similar transforms by scale/rotation/translation, merge nearby groups, rank by `n_matches − RMS_weight × RMS`
5. **Validation** — match_fraction ≥ 12% AND RMS ≤ 1.5 px (optional scale enforcement with `-s`)
6. **WCS fitting** — TAN projection + SIP distortion (degree 2)
7. **Fallback** — Try alternative catalog filters (filter_fallback), optional reflection

**Output**: `*_astro.fits` in `reduced/`. Same WCS applied to all aligned frames of the group.

### 9. Photometric Calibration

**Automatic zeropoint fitting:**

1. **Source detection** — Fixed parameters: 1.2σ threshold, 5 min pixels, 3 px aperture radius. Same iterative SEP + residual re-extraction as astrometry.
2. **Source filtering** — Central 90% of image, isolation ≥ 6 px, ellipticity < 0.4, size within 3σ of stellar locus, SEP flag = 0, exclude VSX variables
3. **Zeropoint** — Weighted mean of `mag_catalog − mag_instrumental` with iterative worst-outlier rejection (target RMS: 0.15 mag)
4. **Limiting magnitude** — `mag_lim = ZP_inst − 2.5·log10(3σ × √(πr²)) + ZP` from median noise in central 80%
5. **Forced photometry** (`-f` flag) — For each target position without a blind detection within the error radius, performs aperture photometry at the exact WCS position. Targets with zero or negative flux are skipped (the epoch becomes a regular non-detection). For positive flux, applies the zeropoint to get a calibrated magnitude and appends to the source catalog with `flag=4`. If the error exceeds `max_forced_mag_err` (default 0.33 mag), the measurement is treated as an upper limit on the detection map.
6. **Diagnostic outputs** — `*_photometry.txt` (source catalog) + `*_photcal.png` (calibration plot) + `*_detections.png`

Photometry runs on both coadd and individual aligned frames. Only JHK filters are calibrated.

### 10. Standard Star Validation

For PROCTYPE=1 observations: find brightest source within 40″ of standard position, compare calibrated magnitude vs 2MASS. Quality: VERY GOOD (<0.05 mag), GOOD (<0.1), MEDIUM (<0.2), POOR (≥0.2).

### 11. Zeropoint Consistency Check

For each science file: compare its catalog-based ZP with the nearest standard star's ZP (corrected for exposure time difference). Results written to `ZP_check` in photometry files.

### 12. Quality Summary

End-of-run aggregate statistics: RMS quality, rejection quality, and ZP_check distributions.

## Output Structure

```
output/
├── tmp/
│   ├── old/                              # Pre-2025 system (DWANGLE)
│   │   ├── file001.fits                  # Prepared raw frames
│   │   ├── OBJECT_OBSID_SUBID_FILTER_sky.fits
│   │   ├── file001_skysub.fits
│   │   ├── file001_skysub_aligned.fits
│   │   └── OBJECT_OBSID_SUBID_FILTER.fits  # Co-add
│   └── new/                              # Post-2025 system (DITHANGL)
├── catalogs/
│   └── catalog_150.1234_-23.4567.csv     # 2MASS + VSX
├── reduced/
│   ├── OBJECT_OBSID_SUBID_FILTER_astro.fits            # WCS-calibrated co-add
│   ├── *_skysub_aligned_astro.fits                      # WCS-calibrated aligned frames
│   ├── *_astro_photometry.txt                           # Source catalog + calibration
│   ├── *_astro_photcal.png                              # Calibration diagnostic plot
│   ├── *_astro_detections.png                            # Detection map (with target overlay if -f)
│   ├── OBJECT_OBSID_SUBID_FILTER_sky.fits               # Sky pattern
│   ├── FLAT_*.fits                                      # Flat fields (pass-through)
│   ├── FOCUS_*.fits                                     # Focus frames (pass-through)
│   ├── pipelog.txt                                      # Processing log
│   └── *.jpg                                            # Preview images (if enabled)
```

Files that fail astrometry are saved without the `_astro` suffix.

### FITS Keywords

| Keyword | Values | Added |
|---------|--------|-------|
| `PROCTYPE` | 0=FLAT, 1=STD, 2=SCI, -1=FOCUS | File prep |
| `PROCSTAT` | 0=raw, 1=reduced | File prep |
| `PSTATSUB` | 1=sky, 2=skysub, 3=aligned, 4=coadd | Processing |
| `DITHID` | 1-5=position, 98=sky, 99=coadd | Processing |
| `INCOMP` | 0=complete, 1=incomplete dither | Co-add |
| `ASTROP` | 0=not processed, 1=failed, 2=success | Astrometry |

Photometric results (ZP, RMS, MagLim) are in `*_photometry.txt` files, not FITS headers.

## Configuration

All parameters in `config.yaml`. Key sections with actual default values:

### Paths & Calibration

```yaml
paths:
  data_folder: data_2026_01

calibration:
  enable_pixel_mask: true
  mask_file: pixel_mask.fits
  enable_flat_correction: true
```

### Detector

```yaml
detector:
  gain: 5.0           # e-/ADU
  read_noise: 25       # e-
  pixel_scale: 1.221   # arcsec/pixel
```

### Sky Subtraction

```yaml
sky_subtraction:
  central_fraction: 0.8
  sigma_clip: 3.0
  noise_median_factor: 1.253

  source_masking:           # Post-median residual cleanup
    enabled: true
    threshold_sigma: 2.0
    min_area: 3
    bw: 64
    bh: 64

  loo_iterative_gauss:      # Thermal arc removal (K-band)
    enabled: true
    n_iter: 3
    base_sigma: 25.0
    sigma_decay: 0.8
    source_sigma: 1.8
    source_grow_base: 10.0
    filter_enable:
      K: true
      J: false
      H: false

grouping:
  max_time_gap_hours: 9.0
```

### Alignment

```yaml
alignment:
  base_angle: 72
  drizzle_pixfrac: 0.8

  refinement:
    enabled: true
    fit_mode: 'similarity'
    min_pixels: 5
    threshold_sigma: 2.0
    max_sources: 50
    pix_tol: 2.0
    min_matches: 4
    min_match_fraction: 0.15
    accept_rms_px: 1.5
    n_refine_iters: 2
    sigma_clip_iters: 3
    sigma_clip_threshold: 3.0
    fallback_to_blind: true
```

### Detection

```yaml
detection:
  min_pixels: [10, 5]
  threshold_sigma: [2.0, 1.2]
  aperture_radius: 3.5
  margin_frac: 0.02
  instrumental_zeropoint: 0.0
  sep_bw: 64
  sep_bh: 64
  sep_fw: 2
  sep_fh: 2
  deblend_nthresh: 64
  deblend_cont: 0.005
  n_iter: 3
  iter_mask_scale: 2.5
```

### Astrometry

```yaml
astrometry:
  min_sources: 3
  n_sources_detected: 25
  n_sources_catalog: 25
  num_quads: 2500
  threshold_code: 0.05
  pix_tol: 2.0
  min_matches_rank: 3
  min_match_fraction: 0.12
  accept_rms_px: 1.5
  top_matches: 0
  print_best_only: true
  scale_min: 0.95
  scale_max: 1.05
  wcs_projection: 'TAN'
  sip_degree: 2
  try_reflection: false
  skip_filters: ["GRISM", "H2"]

  consensus:
    scale_bin: 0.02
    angle_bin: 0.5
    translation_bin: 5
    merge_translation_px: 5.0
    merge_angle_deg: 1.0
    merge_scale: 0.02
    rank_rms_weight: 1.0

  catalog:
    grouping_tolerance_arcmin: 1.0
    radius_arcmin: 10.0
    download_timeout: 30
    download_retries: 3
    download_limit: 10000
    vsx_match_arcsec: 0.5
    default_mag: 99.0
    default_error: 0.4

  filter_fallback:
    J: ["J", "H"]
    K: ["K", "H"]
    H: ["H"]
    H2: ["H"]
    Z: ["H"]
    GRI: ["H"]
```

### Photometry

```yaml
photometry:
  enabled: true
  threshold_sigma: 1.2
  min_pixels: 5
  aperture_radius: 3
  central_fraction: 0.90
  min_isolation_dist: 6.0
  match_tolerance: 2.0
  min_calibration_stars: 3
  max_inst_mag_err: 0.3
  target_rms: 0.15
  max_forced_mag_err: 0.33    # Max error for forced photometry to count as detection
  calibrate_filters: ['J', 'H', 'K']
  standard_check_tolerance_arcsec: 40.0

  star_selection:
    max_ellipticity: 0.4
    size_sigma: 3.0
    max_flag: 0

  quality_thresholds:
    rms:
      very_good: 0.1
      good: 0.175
      medium: 0.25
      poor: 0.35
    rejection:
      good: 0.25
      medium: 0.50
      poor: 0.70
    zp_comparison:
      very_good: 0.05
      good: 0.1
      medium: 0.2
      poor: 0.3
```

### Co-addition

```yaml
coadd:
  sigma_clip: 4.0
  sigma_clip_iters: 2
```

### Detection Map & Preview

```yaml
detection_map:
  enabled: false
  colormap: 'Greys'
  dpi: 200
  figsize: [8, 8]

preview:
  enabled: false
  colormap: 'Greys'
  invert: true
  dpi: 100
```

See `config.yaml` for the complete parameter reference with inline comments.

## Quality Metrics

### Photometric RMS

| Classification | RMS [mag] |
|----------------|-----------|
| VERY GOOD | < 0.10 |
| GOOD | < 0.175 |
| MEDIUM | < 0.25 |
| POOR | < 0.35 |
| VERY POOR | ≥ 0.35 |

### Rejection Fraction

| Classification | Fraction |
|----------------|----------|
| GOOD | < 25% |
| MEDIUM | 25–50% |
| POOR | 50–70% |
| VERY POOR | ≥ 70% |

### ZP Comparison (Standard Stars)

| Classification | Delta ZP |
|----------------|-------------|
| VERY GOOD | < 0.05 |
| GOOD | < 0.10 |
| MEDIUM | < 0.20 |
| POOR | < 0.30 |
| VERY POOR | ≥ 0.30 |

## Troubleshooting

### No sources detected

- Lower `detection.threshold_sigma` (try [1.5, 1.0])
- Reduce `detection.min_pixels` (try [5, 3])
- Check sky subtraction quality (`*_skysub.fits`)

### Astrometry fails

- Remove `-s` flag (scale constraint)
- Increase `astrometry.catalog.radius_arcmin` (try 15–20)
- Enable `try_reflection: true`
- Check catalog download in `catalogs/`

### Photometry fails

- Check filter is in `calibrate_filters` (J, H, K only)
- Verify ASTROP=2 in FITS header
- Increase `photometry.match_tolerance` for poor WCS

### Flat not found

- Filename format: `{FILTER}_dither{ANGLE}_flat.fits`
- Angles: 0, 72, 144, 216, 288
- Check `paths.data_folder` in config

### Thermal arc residual (K band)

Enable LOO + Iterative Gaussian mode:
```yaml
sky_subtraction:
  loo_iterative_gauss:
    enabled: true
    filter_enable:
      K: true
```
This uses leave-one-out sky subtraction (breaking the arc correlation) followed by iterative Gaussian background removal to clean up residuals. See the Configuration section for tuneable parameters.

### Debug Workflow

1. **Run with verbose logging**:
   ```bash
   python remirpipe.py -i data -v 2>&1 | tee debug.log
   ```

2. **Check processing log**:
   ```bash
   less pipelog.txt
   # Search for ERROR, WARNING, or FAILED
   ```

3. **Inspect intermediate files**:
   ```bash
   # Check sky subtraction quality
   ds9 tmp/old/*_skysub.fits
   # or
   ds9 tmp/new/*_skysub.fits
   
   # Check alignment
   ds9 tmp/old/*_aligned.fits
   
   # Check co-add before astrometry
   ds9 tmp/old/*.fits
   ```

4. **Test with single target**:
   ```bash
   # Move single observation to test directory
   python remirpipe.py -i test_obs -co -v
   ```

5. **Start fresh**:
   ```bash
   # Clean all outputs and retry
   python remirpipe.py -i data -co -v
   ```

6. **Check calibration files**:
   ```bash
   # Verify flat fields exist and load correctly
   ls -lh data_2026_01/*_flat.fits
   
   # Check pixel mask
   ds9 data_2026_01/pixel_mask.fits
   ```

## Pipeline Statistics

At completion, the pipeline reports:

- **Files deleted**: DITHID=98,99 from previous runs
- **Sky frames**: Number created
- **Co-adds**: Number generated
- **Astrometry**: Success/failure rate (coadds + aligned frames)
- **Photometry**: Calibrations performed (coadds + aligned frames, JHK only)
- **Attempt breakdown**: Which detection parameters succeeded most often
- **Photometry table**: Per-file zeropoint, RMS, star count, limiting magnitude
- **Standard star checks**: Calibrated vs catalog magnitude comparison
- **Zeropoint consistency**: Catalog-based vs standard-star-based ZP comparison
- **Quality flags**: Distribution of RMS, rejection, and ZP_check quality grades

Example output:
```
=== Pipeline completed! Global summary: ===
  Total coadds processed: 22
  Coadds created: 22
  Sky frames created: 110
  Files deleted during prep: 12
  Astrometrized (coadds+aligned): 120 (95.2%)  |  Failed astrometry: 6 (4.8%)
  Photometrized: 108/120 (90.0%) (JHK only)

Breakdown of successful attempts (thresh-npix-filter-reflection):
   85 x th=2.0-np=10-f=H-ref=0
   35 x th=1.2-np=5-f=H-ref=0
```

## Batch Processing

If you downloaded data spanning multiple nights, the archive unpacks as separate date folders:

```
all_nights/
├── 20260115/
├── 20260116/
├── ...
└── 20260130/
```

Run the pipeline on every night with a single loop:

```bash
cd all_nights/
for dir in */; do
    echo "========== Processing: $dir =========="
    python remirpipe.py -i "$dir" -o "$dir"/proc -s -v -co -t OBJECT_NAME
done
```

A companion notebook (`batch_analyses_and_lightcurve.ipynb`) can then collect all `*_photometry.txt` files across nights and build a time-sorted light curve — see the notebook header for documentation.

## Authors & Credits

**REMIR Pipeline** - Automated reduction for REM telescope near-infrared data

**Dependencies:**
- [Astropy](https://www.astropy.org/) - Astronomy fundamentals
- [Photutils](https://photutils.readthedocs.io/) - Aperture photometry
- [SEP](https://sep.readthedocs.io/) - Source Extraction and Photometry
- [NumPy](https://numpy.org/) - Numerical computing
- [SciPy](https://scipy.org/) - Scientific computing
- [Matplotlib](https://matplotlib.org/) - Plotting
- [PyYAML](https://pyyaml.org/) - YAML parsing
- [Pandas](https://pandas.pydata.org/) - Data manipulation

**Catalogs:**
- [2MASS Point Source Catalog](https://irsa.ipac.caltech.edu/Missions/2mass.html) - Astrometry & photometry reference
- [AAVSO International Variable Star Index (VSX)](https://www.aavso.org/vsx/) - Variable star identification

**Algorithm References:**
- Drizzling: Fruchter & Hook 2002 ([PASP 114:144](https://ui.adsabs.harvard.edu/abs/2002PASP..114..144F))
- Quad matching: Lang et al. 2010 ([AJ 139:1782](https://ui.adsabs.harvard.edu/abs/2010AJ....139.1782L), Astrometry.net)
- Source extraction: Bertin & Arnouts 1996 ([A&AS 117:393](https://ui.adsabs.harvard.edu/abs/1996A%26AS..117..393B), SExtractor)

## License

[Specify license here]

## Version History

**v2.2** (2026-03)
- **Shift-only alignment refinement** (`fit_mode: 'shift'`)
  - Sigma-clipped median of matched coordinate differences (2 parameters: dx, dy)
  - Default remains `fit_mode: 'similarity'` (full similarity transform, 4 parameters)
  - Use `'shift'` when inter-frame rotation is negligible (e.g., REMIR dithers <0.02°)
- **Alignment rescue mechanism**: when initial cross-match fails (<15% matches), automatically retries with progressively wider tolerance (2×, 4×, 8×), recovering frames that previously fell back to inaccurate blind shifts

**v2.1** (2026-03)
- **LOO + Iterative Gaussian sky subtraction mode** (`loo_iterative_gauss.enabled: true`)
  - Leave-one-out sky (median of N−1 other frames) breaks wedge-prism arc correlation
  - 3-iteration coarse-to-fine Gaussian background subtraction removes residual arc pattern
  - Full noise propagation through LOO sky, flat division, and IterGauss background subtraction
  - Fully configurable: iteration count, sigma schedule, source masking parameters
  - Standard N-frame sky still saved as product for output consistency
  - Activated via single config toggle; all parameters tuneable in `config.yaml`
  - **Normalized convolution** for source-masked background estimation (replaces global median fill), eliminating negative halos around bright sources
  - **Fixed source mask** built once before the iteration loop (replaces per-iteration re-detection), eliminating negative "holes" in the coadd caused by mask inconsistencies

**v2.0** (2026-02)
- Median sky subtraction from all N frames (simpler, more robust)
- **Single-pass affine drizzle** for alignment: translation or full rotation + scale + translation
  in one drizzle step, no double interpolation (same principle as AQuA/PREPROCESS)
- Processing order: level → sky → flat
- Optional alignment refinement via cross-matching (shift-only or similarity transform)
- Fixed noise model (read noise in correct ADU² units)
- Enhanced FITS keyword tracking
- Comprehensive documentation

**v1.0** (2026-01)
- Initial release
- Basic calibration pipeline
- Quad-matching astrometry
- 2MASS photometric calibration

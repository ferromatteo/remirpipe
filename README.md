# REMIR Pipeline

Complete automated reduction and analysis pipeline for REMIR (REM InfraRed) near-infrared imaging data.

## Overview

The REMIR pipeline performs end-to-end processing of NIR imaging data from raw FITS files through calibration, alignment, co-addition, astrometric calibration, and photometric calibration. Designed for dithered observations in J, H, and K bands with the REM telescope's rotating wedge prism dither system.

**Key Features:**
- Median sky subtraction from all dithered frames
- Thermal pattern correction (EXPTIME-proportional scaling)
- Drizzling algorithm for flux-preserving alignment
- Inverse-variance weighted co-addition with optimal noise propagation
- Quad-matching astrometry with 2MASS catalog
- Automated photometric calibration and quality assessment

## Requirements

### Python Dependencies

```bash
pip install numpy scipy astropy photutils matplotlib pyyaml pandas sep requests
```

**Required packages:**
- `numpy` ≥ 1.20 - Array operations and linear algebra
- `scipy` ≥ 1.7 - Spatial operations (cKDTree for matching)
- `astropy` ≥ 5.0 - FITS I/O, WCS, coordinates, time handling
- `photutils` ≥ 1.5 - Aperture photometry
- `sep` ≥ 1.2 - Source Extraction and Photometry (SExtractor in Python)
- `matplotlib` ≥ 3.5 - Diagnostic plots and preview generation
- `pyyaml` ≥ 6.0 - Configuration file parsing
- `pandas` ≥ 1.3 - Catalog handling and table operations
- `requests` - HTTP catalog downloads (2MASS, VSX)

### Calibration Files

Create calibration files following [CALIBRATION_CREATION_GUIDE.md](CALIBRATION_CREATION_GUIDE.md).

Place in `data_folder` directory (configured in `config.yaml`, default: `data_2026_01/`):

```
data_2026_01/
├── pixel_mask.fits                          # Bad pixel mask (0=bad, 1=good)
├── J_dither0_flat.fits                      # Master flat fields
├── J_dither72_flat.fits                     # (per filter × dither angle)
├── J_dither144_flat.fits
└── ...
```

**Required:**
- Pixel mask: `pixel_mask.fits`
- Flats: `{FILTER}_dither{ANGLE}_flat.fits` for each filter (J/H/K) × dither (0/72/144/216/288)

## Installation

1. Clone or download repository:
```bash
git clone <repository_url>
cd remir-pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare calibration files (see `CALIBRATION_CREATION_GUIDE.md`)

4. Edit `config.yaml` to match your setup (data paths, processing parameters)

## Usage

### Basic Command

```bash
python remirpipe.py -i /path/to/raw/data -o /path/to/output -v
```

### Command-Line Options

```
Required:
  -i, --input DIR          Input directory with raw FITS files

Optional:
  -o, --output DIR         Output directory (default: same as input)
  -c, --config FILE        Configuration file (default: config.yaml)
  -v, --verbose            Verbose output to console and log
  -d, --delete-tmp         Delete tmp/ directory after completion
  -s, --scale-constraint   Apply strict scale limits (0.95-1.05) for astrometry
  -co, --clean-output      Clean existing output directories before starting
```

### Examples

**Standard processing with verbose logging:**
```bash
python remirpipe.py -i ./raw_data/2026-01-15 -v
```

**Custom configuration and clean start:**
```bash
python remirpipe.py -i ./data -o ./reduced -c custom.yaml -co -v
```

**Strict astrometry with cleanup:**
```bash
python remirpipe.py -i ./data -s -d -v
```

## Pipeline Workflow

The pipeline executes these steps automatically:

### 1. File Preparation

- **Decompress**: Gunzip `.fits.gz` files in-place
- **Filter**: Remove files with DITHID=98 or DITHID=99 (pipeline products from previous runs)
- **Fix headers**: Repair invalid values (e.g., NaN in WINDDIR)
- **Add keywords**: FILENAME, PROCTYPE (0=FLAT, 1=STD, 2=SCI, -1=FOCUS)
- **Mask bad pixels**: Apply pixel_mask.fits (sets masked pixels to NaN)

### 2. File Classification

- Classify by dither system: **old** (pre-2025, DWANGLE) vs **new** (post-2025, DITHANGL)
- Route files:
  - FLAT/FOCUS → directly to `reduced/`
  - SCI/STD → to `tmp/old/` or `tmp/new/` for processing

### 3. Grouping & Validation

- Group by: OBJECT + FILTER + OBSID + SUBID + time gap < 9 hours (configurable)
- Validate dither completeness:
  - **Complete**: N files = NDITHERS (typically 5)
  - **Incomplete**: 3 ≤ N < NDITHERS (processable)
  - **Defective**: N < 3 or N > NDITHERS (skipped)

### 4. Calibration & Sky Subtraction

**Processing order** (per group of N dithered frames):

1. **Load raw masked data**
   - All frames already have pixel mask applied during file preparation

2. **Level normalization**
   - For each frame *i*: compute *median* of central 80% region (σ-clipped)
   - Calculate mean of all medians: `mean_level = mean([med[0], med[1], ..., med[N-1]])`
   - Scale all frames: `data_leveled[i] = data_raw[i] × (mean_level / med[i])`
   - Scale noise identically: `noise_leveled[i] = noise_raw[i] × (mean_level / med[i])`
   - **Purpose**: Equalize background levels before sky creation (removes large-scale variations)

3. **Single sky pattern creation**
   - Use ALL N leveled frames (not leave-one-out)
   - Compute pixel-wise median: `sky_pattern = median([data_leveled[0], ..., data_leveled[N-1]])`
   - Save: `{OBJECT}_{OBSID}_{SUBID}_{FILTER}_sky.fits` (DITHID=98, PSTATSUB=1)

4. **Sky subtraction and flat fielding**
   - Sky subtract: `data_skysub[i] = data_leveled[i] - sky_pattern`
   - Load flat matching (filter, dither angle)
   - Apply flat: `data_final[i] = data_skysub[i] / flat[i]`
   - Save: `originalname[i]_skysub.fits` (DITHID=original, PSTATSUB=2)

5. **Thermal residual correction** (post-processing, linear EXPTIME scaling)
   - Group all skysub files by (filter, dither_angle)
   - **Diversity gate** (must pass both to proceed):
     - ≥ 10 files in the group (configurable `thermal_requirements.min_files`)
     - ≥ 3 unique sky pointings separated by ≥ 10″ each (configurable `min_positions` / `min_separation_arcsec`)
   - Scale each file to 10 s reference: `data_scaled = data × (10 / EXPTIME)`
   - Create template: `thermal_template = median(data_scaled)` (pattern at 10 s)
   - Zero-center: `template_centered = template - median(template)`
   - For each file: `α = EXPTIME / 10`
   - Apply: `corrected = data − α × template`
   - **THMALPHA** keyword records the scaling factor (`EXPTIME / 10`)

**Processing formula**: `data_final = (data_raw × level_factor - sky_all) / flat - α × thermal`

### 5. Alignment

- **Geometric shifts**: Calculate from dither wedge angle + calibrated parameters
- **Optional refinement**: Cross-match sources between frames, fit **similarity transform** (rotation + scale + translation) using least-squares with:
  - **Flux-weighted fitting**: bright stars (good centroids) dominate the fit
  - **3σ sigma-clipping**: outlier matches rejected iteratively
  - **Iterative re-matching**: after first fit, re-project sources through the transform and re-match with tighter tolerance, recovering borderline matches
  - **Source capping**: only brightest 50 sources used (avoids noise from faint detections)
  - SEP deblending overflow handled gracefully (`set_sub_object_limit(4096)`)
- **Single-pass affine drizzle**:
  - Full rotation + scale + translation applied in one drizzle pass (no double interpolation)
  - Flux-conserving alignment based on geometric pixel overlap
  - Same principle as AQuA/PREPROCESS's polygon-intersection remapping
  - Preserves image sharpness (no interpolation smoothing kernel)
  - Configurable pixfrac (default: 0.8 for ~10% sharper PSF)
- Save aligned frames: `originalname[i]_skysub_aligned.fits` (PSTATSUB=3)

### 6. Co-addition

- **Inverse-variance weighted mean** of aligned frames: `coadd = Σ(data_i / σ_i²) / Σ(1 / σ_i²)`
- **Optimal noise**: `noise = √(1 / Σ(1 / σ_i²))`
- Drizzle-edge pixels with higher noise are automatically down-weighted
- No outlier rejection is applied (cosmic rays may propagate)
- **Output**: `OBJECT_OBSID_SUBID_FILTER.fits` (DITHID=99, PSTATSUB=4), includes WEIGHT extension
- **Header keywords**:
  - `EXPTIME`: Total exposure time (sum of all frames)
  - `NCOADD`: Number of coadded frames
  - `INCOMP`: 1 if incomplete dither sequence, 0 if complete

**S/N improvement**: √N for N frames (optimal via inverse-variance weighting)

### 7. Catalog Download

- Query **2MASS Point Source Catalog** via INAF service:
  - Cone search around image center (default: 10 arcmin radius)
  - Columns: RAJ2000, DEJ2000, Jmag, Hmag, Kmag + errors
  - Exponential backoff retry on network failures
- Query **VSX** (Variable Star Index):
  - Same cone search
  - Cross-match variables with 2MASS (0.5" tolerance)
- Cache catalogs: `catalogs/catalog_{RA}_{DEC}.csv`
- **Grouping**: Images within 1 arcmin share same catalog (efficient)

### 8. Astrometric Calibration

**Quad-matching algorithm**:

1. **Source detection** (SEP/SExtractor backend)
   - Try multiple parameter combinations: min_pixels=[10,5], threshold=[2.0,1.2]
   - Extract brightest N sources (default: 25)

2. **Geometric quad generation**
   - Build ~2500 quads from detected sources
   - Build ~2500 quads from catalog sources
   - Encode quad geometry as hash (4 invariant codes)

3. **Quad matching**
   - Match detection quads → catalog quads by geometric similarity
   - Threshold: 0.05 (configurable)

4. **Transform consensus**
   - Group similar transforms by scale/rotation/translation
   - Merge nearby groups
   - Select largest consensus group

5. **Validation**
   - Check: match_fraction ≥ 12% AND RMS ≤ 1.5 pixels
   - Optional: enforce scale ∈ [0.95, 1.05] with `-s` flag

6. **WCS fitting**
   - Fit TAN projection + SIP distortion (degree 2)
   - Update FITS header with WCS keywords

7. **Fallback**
   - Try reflection (flip image) if normal orientation fails
   - Try alternative catalog filters (e.g., H-band for J-band image)

8. **Filter skipping**
   - Filters in `skip_filters` list (e.g., GRISM, H2) bypass astrometry entirely
   - These coadds and their aligned frames are copied to `reduced/` as-is

**Output**: WCS-calibrated FITS in `reduced/` directory

**Note**: The same WCS solution is applied to both the coadd and all its individual aligned frames. Each aligned frame is also saved to `reduced/` with the coadd's WCS.

### 9. Photometric Calibration

**Automatic zeropoint fitting**:

1. **Source detection** (fixed parameters)
   - Threshold: 1.2σ, min_pixels: 5
   - Aperture photometry: 3.0 pixel radius

2. **Source filtering**
   - Keep only central 90% of image (avoid edge effects)
   - Reject crowded sources (min separation: 8 pixels)
   - Match to 2MASS catalog (tolerance: 2 pixels in WCS space)

3. **Zeropoint calculation**
   - For each matched star: `ZP_i = mag_catalog - mag_instrumental`
   - Iterative 3σ clipping to reject outliers
   - Final: `ZP = median(ZP_i)`, `RMS = std(ZP_i)`

4. **Quality assessment**
   - Minimum 3 stars required
   - RMS quality: VERY GOOD (<0.1), GOOD (<0.175), MEDIUM (<0.25), POOR (<0.35)
   - Rejection quality: fraction of stars rejected

5. **Limiting magnitude**
   - 3σ detection limit in instrumental mags → calibrated mags

6. **Diagnostic outputs**
   - Photometry catalog: `*_photometry.txt` (all detected sources + matches)
   - Diagnostic plot: `*_photometry.png` (instrumental vs catalog)
   - VSX variables flagged in catalog

**Note**: Photometry is run independently on **both** the coadd and each individual aligned frame. This provides per-frame zeropoints useful for monitoring and quality control.

**Header keywords added**:
- `ZP_{filter}`: Photometric zeropoint [mag]
- `ZP_ERR`: Zeropoint uncertainty (RMS) [mag]
- `ZP_NSTAR`: Number of stars used for calibration
- `ZP_RMSMG`: RMS of fit [mag]
- `ZP_QUAL`: Quality classification
- `MAG_LIM`: 3σ limiting magnitude [mag]
- `PHOTSTAT`: Photometry status (0=failed, 1=success)

**Only JHK filters** are photometrically calibrated (configured via `calibrate_filters`). Filters without 2MASS data (Z, GRI, H2) are skipped.

### 10. Standard Star Validation

For PROCTYPE=1 (standard star observations):
- Find brightest detection within tolerance radius of the standard's RA/DEC
- Compare calibrated magnitude vs 2MASS catalog magnitude
- Calculate offset: ΔZP = |mag_calibrated - mag_catalog|
- Quality: VERY GOOD (<0.05), GOOD (<0.1), MEDIUM (<0.2), POOR (≥0.2)
- Flag inconsistencies in log

### 11. Zeropoint Consistency Check

- Cross-check catalog-based zeropoint against standard-star-based zeropoint
- Compare ZP from field star calibration vs ZP derived from standard star measurement
- Quality assessment using same thresholds as standard star validation
- Updates `ZP_CHECK` keyword in photometry output files

### 12. Quality Flags Summary

At the end of processing, the pipeline prints aggregate statistics:
- **RMS quality distribution**: counts of VERY GOOD / GOOD / MEDIUM / POOR / VERY POOR
- **Rejection quality distribution**: counts of GOOD / MEDIUM / POOR
- **ZP_check quality distribution**: counts from standard star and consistency checks

## Output Structure

```
output/
├── tmp/                                    # Temporary processing files
│   ├── old/                                # Pre-2025 system (DWANGLE) - all products
│   │   ├── file001.fits                  # Raw frames (DITHID=1-5, PSTATSUB=0)
│   │   ├── OBJECT_OBSID_SUBID_FILTER_sky.fits  # Single sky per group (DITHID=98, PSTATSUB=1)
│   │   ├── file001_skysub.fits           # Sky-subtracted (DITHID=1-5, PSTATSUB=2)
│   │   ├── file001_skysub_aligned.fits   # Aligned (DITHID=1-5, PSTATSUB=3)
│   │   ├── OBJECT_OBSID_SUBID_FILTER.fits  # Co-add (DITHID=99, PSTATSUB=4)
│   │   └── ...
│   └── new/                                # Post-2025 system (DITHANGL) - all products
│       ├── file002.fits
│       ├── OBJECT_OBSID_SUBID_FILTER_sky.fits
│       ├── file002_skysub.fits
│       ├── file002_skysub_aligned.fits
│       ├── OBJECT_OBSID_SUBID_FILTER.fits
│       └── ...
├── catalogs/                              # Downloaded reference catalogs
│   ├── catalog_150.1234_-23.4567.csv     # 2MASS + VSX data
│   └── ...
├── reduced/                               # Final calibrated products
│   ├── OBJECT_OBSID_SUBID_FILTER.fits    # WCS-calibrated co-adds
│   ├── *_skysub_aligned.fits              # WCS-calibrated aligned frames (same WCS as coadd)
│   ├── OBJECT_OBSID_SUBID_FILTER_photometry.txt  # Source catalogs (coadd + per-frame)
│   ├── OBJECT_OBSID_SUBID_FILTER_photometry.png  # Diagnostic plots (coadd + per-frame)
│   ├── OBJECT_OBSID_SUBID_FILTER_sky.fits # Sky pattern (DITHID=98)
│   ├── FLAT_*.fits                        # Flat fields (pass-through)
│   ├── FOCUS_*.fits                       # Focus frames (pass-through)
│   ├── pipelog.txt                        # Complete processing log
│   └── *.jpg                              # Preview images (if enabled)
```

### File Type Markers (FITS Keywords)

| File Type | DITHID | PSTATSUB | Location |
|-----------|--------|----------|----------|
| Raw science | 1-5 | 0 | tmp/old/ or tmp/new/ |
| Sky pattern | 98 | 1 | tmp/old/ or tmp/new/ |
| Sky-subtracted | 1-5 | 2 | tmp/old/ or tmp/new/ |
| Aligned | 1-5 | 3 | tmp/old/ or tmp/new/ |
| Co-add | 99 | 4 | tmp/old/ or tmp/new/, reduced/ |
| Flat field | varies | 1 | reduced/ |
| Focus frame | varies | 1 | reduced/ |

**Note**: All intermediate products (raw, sky, skysub, aligned, coadd) are stored together in `tmp/old/` or `tmp/new/` based on dither system. Sky patterns have DITHID=98 (generic sky marker). Each group produces one sky pattern shared by all N frames.

All processing parameters in `config.yaml`. Key sections:

### Paths & Calibration

```yaml
paths:
  data_folder: data_2026_01    # Calibration files directory

calibration:
  enable_pixel_mask: true      # Apply bad pixel mask
  mask_file: pixel_mask.fits
  
  enable_flat_correction: true # Flat fielding (applied after sky subtraction)
  
  enable_thermal_correction: true   # Thermal residual correction (fringing-style)
  thermal_filters: [K]               # Filters for thermal correction
  thermal_requirements:
    min_files: 10              # Min skysub files per (filter, dither_angle) group
    min_positions: 3           # Min unique RA/DEC pointings
    min_separation_arcsec: 10.0  # Two pointings are "different" if ≥ this apart
```

### Detector Parameters

```yaml
detector:
  gain: 5.0                    # e-/ADU (REMIR nominal)
  read_noise: 25               # e- (NICS RRR mode)
  pixel_scale: 1.221           # arcsec/pixel
```

### Sky Subtraction

```yaml
sky_subtraction:
  central_fraction: 0.8        # Region for median calculation
  sigma_clip: 3.0              # Outlier rejection threshold
  noise_median_factor: 1.253   # Noise scaling for median
```

### Alignment

```yaml
alignment:
  base_angle: 72               # Base rotation offset [degrees]
  drizzle_pixfrac: 0.8         # Pixel fraction (0.8 = sharper PSF, 1.0 = full pixel)
  
  refinement:
    enabled: true              # Enable similarity-transform refinement
    min_pixels: 5              # Detection threshold
    threshold_sigma: 2.0
    max_sources: 50            # Cap to N brightest
    min_matches: 4             # Min matched sources
    accept_rms_px: 1.5         # Max RMS [pixels]
    n_refine_iters: 2          # Re-match iterations
    sigma_clip_iters: 3        # Sigma-clipping rounds
    sigma_clip_threshold: 3.0  # Rejection threshold [sigma]
  
  old:  # Pre-2025 system parameters
    dithangl_key: DWANGLE
    theta_offset: -72
    theta_n: 5
    r_n: 17
  
  new:  # Post-2025 system parameters
    dithangl_key: DITHANGL
    theta_offset: 0
    theta_n: 5
    r_n: 17
```

### Co-addition

*(No configurable parameters — inverse-variance weighted mean is always used)*

### Detection & Astrometry

```yaml
detection:
  min_pixels: [10, 5]          # Try in order
  threshold_sigma: [2.0, 1.2]  # Try in order
  aperture_radius: 3.5         # Photometry aperture [pixels]

astrometry:
  min_sources: 3               # Min sources to attempt
  n_sources_detected: 25       # N brightest for quads
  n_sources_catalog: 25
  num_quads: 2500              # Max quads to generate
  threshold_code: 0.05         # Geometric matching threshold
  
  min_match_fraction: 0.12     # Min fraction matched
  accept_rms_px: 1.5           # Max RMS [pixels]
  
  catalog:
    radius_arcmin: 10.0        # Cone search radius
    download_timeout: 30       # HTTP timeout [sec]
  
  filter_fallback:             # Try filters in order
    J: ["J", "H"]
    K: ["K", "H"]
    H: ["H"]
    H2: ["H"]
    Z: ["H"]
  
  skip_filters: ["GRISM", "H2"]  # Filters to skip astrometry entirely
  try_reflection: false          # Try mirror-flipped geometry if normal fails

photometry:
  enabled: true
  aperture_radius: 3.0         # Fixed aperture [pixels]
  central_fraction: 0.90       # Central region only
  min_isolation_dist: 8.0      # Min separation [pixels]
  match_tolerance: 2.0         # Max match distance [pixels]
  sigma_clip: 3.0              # Outlier rejection
  min_calibration_stars: 3     # Min stars required
  max_inst_mag_err: 0.2        # Max instrumental mag error
  target_rms: 0.2              # Target RMS for clipping [mag]
  calibrate_filters: ['J', 'H', 'K']  # Only these have 2MASS calibration
```

**See `config.yaml` for complete parameter documentation with inline comments.**

## FITS Header Keywords

### Standard Keywords (Input)

| Keyword | Description | Values |
|---------|-------------|--------|
| `OBJECT` | Target name | String |
| `FILTER` | Filter name | J, H, K, H2, etc. |
| `EXPTIME` | Exposure time | Seconds |
| `OBSTYPE` | Observation type | FLATF, STDSTAR, etc. |
| `IMAGETYP` | Image type | OBJECT, FOCUS |
| `OBSID` | Observation ID | Integer |
| `SUBID` | Sub-observation ID | Integer |
| `NDITHERS` | Expected dither positions | Integer (typically 5) |
| `DITHANGL` | Dither angle (new system) | Degrees (0-360) |
| `DWANGLE` | Dither wedge angle (old system) | Degrees |
| `DATE-OBS` | Observation timestamp | ISO format |

### Pipeline-Added Keywords

| Keyword | Description | Values | Added When |
|---------|-------------|--------|------------|
| `PROCTYPE` | Processing type | 0=FLAT, 1=STD, 2=SCI, -1=FOCUS | File prep |
| `PROCSTAT` | Processing status | 0=raw, 1=reduced | File prep / processing |
| `PSTATSUB` | Processing sub-status | 1=sky, 2=skysub, 3=aligned, 4=coadd | Processing |
| `DITHID` | Dither ID | 1-5=position, 98=sky, 99=coadd | Processing |
| `FILENAME` | File name | String | File prep |
| `INCOMP` | Incomplete dither flag | 0=complete, 1=incomplete | Co-add |
| `THMALPHA` | Thermal correction scaling | Float (optimal α) | Thermal |
| `ASTROP` | Astrometry processed | 0=no, 1=yes | Astrometry |
| `ZP_{filter}` | Photometric zeropoint | Magnitude | Photometry |
| `ZP_ERR` | Zeropoint RMS | Magnitude | Photometry |
| `ZP_NSTAR` | Stars used for ZP | Integer | Photometry |
| `ZP_RMSMG` | RMS of ZP fit | Magnitude | Photometry |
| `ZP_QUAL` | Zeropoint quality | String | Photometry |
| `MAG_LIM` | 3σ limiting magnitude | Magnitude | Photometry |
| `PHOTSTAT` | Photometry status | 0=failed, 1=success | Photometry |

### WCS Keywords (Updated by Astrometry)

Standard WCS + SIP distortion keywords added by astropy WCS fitting.

## Quality Metrics

### Photometric RMS Quality

| Classification | RMS Threshold | Interpretation |
|----------------|---------------|----------------|
| **VERY GOOD** | < 0.10 mag | Excellent calibration |
| **GOOD** | < 0.175 mag | Good calibration |
| **MEDIUM** | < 0.25 mag | Acceptable |
| **POOR** | < 0.35 mag | Marginal |
| **VERY POOR** | ≥ 0.35 mag | Questionable |

### Rejection Quality

| Classification | Rejection Fraction | Interpretation |
|----------------|-------------------|----------------|
| **GOOD** | < 25% | Clean star field |
| **MEDIUM** | 25-50% | Some contamination |
| **POOR** | ≥ 50% | Crowded/problematic |

### Standard Star Zeropoint Comparison

| Classification | |ΔZP| Threshold | Interpretation |
|----------------|---------------|----------------|
| **VERY GOOD** | < 0.05 mag | Excellent agreement |
| **GOOD** | < 0.10 mag | Good agreement |
| **MEDIUM** | < 0.20 mag | Acceptable |
| **POOR** | ≥ 0.20 mag | Check calibration |

## Troubleshooting

### Common Issues

#### No sources detected
**Symptoms**: Astrometry fails with "Not enough sources detected"

**Solutions**:
- Lower `threshold_sigma` in config (try 1.0-1.2)
- Reduce `min_pixels` (try 3-5)
- Check sky subtraction quality (look at `*_skysub.fits`)
- Verify image is not saturated or very faint

#### Astrometry fails repeatedly
**Symptoms**: All quad-matching attempts fail

**Solutions**:
- Disable scale constraint (remove `-s` flag)
- Check initial WCS in header (RA, DEC, CRPIX, CDELT)
- Increase `catalog_radius_arcmin` (try 15-20)
- Enable reflection: `try_reflection: true` in config
- Verify catalog download successful (check `catalogs/`)
- Try different `filter_fallback` (use H-band for all)

#### Photometry calibration fails
**Symptoms**: "Photometry failed" or PHOTSTAT=0

**Solutions**:
- Check filter in `calibrate_filters` list (J, H, K only)
- Verify astrometry succeeded (ASTROP=1)
- Lower `min_calibration_stars` (try 2-3 for sparse fields)
- Increase `match_tolerance` (try 3.0-5.0 for poor WCS)
- Check 2MASS catalog has sufficient stars at this position

#### Flat field not found
**Symptoms**: "No flat found for filter X"

**Solutions**:
- Check filename format: `{FILTER}_dither{ANGLE}_flat.fits`
- Verify filter name is uppercase: J, H, K (not j, h, k)
- Confirm `data_folder` path in config is correct
- Dither angles must be rounded to: 0, 72, 144, 216, 288
- Create missing flats (see CALIBRATION_CREATION_GUIDE.md)

#### Thermal correction not applied
**Symptoms**: No thermal correction messages in log

**Solutions**:
- Check `enable_thermal_correction: true` in config
- Verify filter in `thermal_filters` list (e.g., `[K]`)
- Ensure sufficient skysub files for (filter, dither_angle) group (need ≥10, configurable via `thermal_requirements.min_files`)
- Ensure files span ≥3 distinct sky pointings separated by ≥10″ (configurable via `thermal_requirements.min_positions` / `min_separation_arcsec`)
- Check dither angles are correctly rounded to 0, 72, 144, 216, 288

#### Co-add has artifacts
**Symptoms**: Residual cosmic rays or streaks in co-add

**Solutions**:
- No sigma-clipping is currently implemented in co-addition
- Check individual skysub frames for cosmic rays or bad pixels
- Verify alignment quality (check residuals in aligned frames)
- Improve bad pixel mask coverage
- Check for bad individual frames in `tmp/old/` or `tmp/new/`

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

## Performance

Typical processing times (approximate, single-core):

| Step | Time per... | Notes |
|------|-------------|-------|
| File preparation | ~0.5-1 sec/file | Gunzip + header fixes |
| Sky subtraction | ~3-5 sec/group | N=5 dithers |
| Alignment | ~2-3 sec/coadd | Drizzling algorithm |
| Co-addition | ~1-2 sec/coadd | Inverse-variance weighted |
| Astrometry | ~5-30 sec/coadd | Varies with attempts |
| Photometry | ~2-5 sec/coadd | Source detection + matching |

**Total**: ~30-90 seconds per observation block (5 dithers → 1 co-add)

**Batch processing**: Parallelization not implemented; processes sequentially

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

## Authors & Credits

**REMIR Pipeline** - Automated reduction for REM telescope near-infrared data

**Dependencies:**
- [Astropy](https://www.astropy.org/) - Astronomy fundamentals
- [Photutils](https://photutils.readthedocs.io/) - Photometry toolkit (includes SEP)
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

**v2.0** (2026-02)
- Median sky subtraction from all N frames (simpler, more robust)
- **Single-pass affine drizzle** for alignment: full rotation + scale + translation
  in one drizzle step, no double interpolation (same principle as AQuA/PREPROCESS)
- Thermal correction with EXPTIME-based linear scaling
- Processing order: level → sky → flat → thermal
- Optional alignment refinement via cross-matching + similarity transform
- Fixed noise model (read noise in correct ADU² units)
- Enhanced FITS keyword tracking
- Comprehensive documentation

**v1.0** (2026-01)
- Initial release
- Basic calibration pipeline
- Quad-matching astrometry
- 2MASS photometric calibration

---

**Quick Start**: See [CALIBRATION_CREATION_GUIDE.md](CALIBRATION_CREATION_GUIDE.md) for creating calibration files.

**Configuration**: See inline comments in `config.yaml` for detailed parameter descriptions.

**Support**: Check `pipelog.txt` for detailed processing information and error messages.

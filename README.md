# REMIR Pipeline

A complete astronomical image reduction and analysis pipeline for the **REM (Rapid Eye Mount) telescope's infrared camera**. This pipeline processes raw FITS images through calibration, sky subtraction, alignment, co-addition, astrometric calibration, and photometric analysis.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Pipeline Workflow](#pipeline-workflow)
  - [Step 1: File Preparation](#step-1-file-preparation)
  - [Step 2: File Classification](#step-2-file-classification)
  - [Step 3: Sky Subtraction](#step-3-sky-subtraction)
  - [Step 4: Thermal Pattern Correction](#step-4-thermal-pattern-correction)
  - [Step 5: Frame Alignment](#step-5-frame-alignment)
  - [Step 6: Co-addition](#step-6-co-addition)
  - [Step 7: Catalog Download](#step-7-catalog-download)
  - [Step 8: Astrometric Calibration](#step-8-astrometric-calibration)
  - [Step 9: Photometric Calibration](#step-9-photometric-calibration)
- [Output Files](#output-files)
- [Quality Control](#quality-control)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)

---

## Overview

The **REMIR Pipeline** (`remirpipe.py`) is designed to process near-infrared (NIR) astronomical images from the REM telescope's infrared camera. It automates the entire reduction process from raw data to calibrated, science-ready images with accurate astrometry and photometry.

The pipeline supports multiple filters (J, H, K bands) and handles both "old" and "new" camera systems. It performs sophisticated operations including:

- Automated flat-field calibration
- Sky background subtraction using median-combining
- Thermal pattern removal
- Cross-correlation-based frame alignment
- Robust co-addition with sigma clipping
- Astrometric calibration using geometric quad matching
- Photometric calibration against 2MASS catalog
- Comprehensive quality assessment

---

## Features

### Core Capabilities

- **Automatic calibration**: Applies flat-field corrections based on filter and dither angle
- **Sky subtraction**: Creates master sky frames from median-combined groups
- **Thermal correction**: Removes detector thermal patterns using median filtering
- **Frame alignment**: Sub-pixel alignment using cross-correlation with reference selection
- **Image co-addition**: Combines aligned frames with sigma-clipped mean for optimal signal-to-noise
- **Astrometric solution**: Geometric quad-matching algorithm with multiple fallback strategies
- **Photometric calibration**: Zero-point determination using 2MASS catalog stars
- **Quality metrics**: RMS, rejection rate, and zero-point consistency checks
- **Standard star verification**: Validates calibration using standard star observations

### Data Handling

- **Format support**: FITS files, including gzipped (.fits.gz)
- **Filter support**: J, H, K bands
- **System compatibility**: Handles both "old" and "new" camera systems
- **Multiple observation types**: Science (OBJECT), standard stars (STDSTAR), flats (FLATF), focus frames
- **Incomplete groups**: Processes groups with missing frames or calibration data

### Output Products

- Calibrated, sky-subtracted individual frames (`*_c.fits`)
- Aligned frames (`*_c_a.fits`)
- Co-added images (`coadd_*.fits`)
- Sky frames (`sky_*.fits`)
- Astrometrically calibrated images (updated WCS in headers)
- Photometric catalogs (embedded in FITS extensions)
- Calibration plots (zero-point fitting)
- Preview JPEG images for visual inspection
- Comprehensive processing log (`pipelog.txt`)

---

## Requirements

### Python Version

- Python 3.7 or higher

### Dependencies

The pipeline requires the following Python packages:

```
numpy
scipy
astropy
photutils
sep (Source Extractor Python)
pandas
matplotlib
pyyaml
requests
```

### External Services

- **INAF/2MASS catalog service**: Used for downloading reference catalogs (requires internet connection)

### Install Dependencies

```bash
pip install numpy scipy astropy photutils sep pandas matplotlib pyyaml requests
```

Or if using conda:

```bash
conda install numpy scipy astropy photutils pandas matplotlib pyyaml requests
pip install sep
```

---

## Installation

1. Clone or download the repository containing `remirpipe.py`

2. Ensure all dependencies are installed (see [Requirements](#requirements))

3. Create a `config.yaml` file (see [Configuration](#configuration))

4. Make the pipeline executable (optional):

```bash
chmod +x remirpipe.py
```

---

## Configuration

The pipeline is configured via a YAML file (`config.yaml`). By default, it looks for `config.yaml` in the same directory as the script.

### Configuration File Structure

The configuration file contains the following main sections:

#### 1. Detection Parameters

Controls source detection for astrometry and photometry:

```yaml
detection:
  min_pixels: [5, 7, 10]  # Minimum connected pixels for source detection
  threshold_sigma: [2.0, 2.5, 3.0]  # Detection threshold in sigma above background
  aperture_radius: 8.0  # Aperture radius in pixels for photometry
  margin_frac: 0.02  # Edge margin fraction for catalog matching
```

#### 2. Astrometry Parameters

Controls astrometric calibration:

```yaml
astrometry:
  num_quads: 100  # Number of geometric quads to build
  n_sources_detected: 25  # Top N brightest detected sources to use
  n_sources_catalog: 25  # Top N brightest catalog sources to use
  min_sources: 3  # Minimum sources required to attempt astrometry
  top_matches: 50  # Number of top quad matches to evaluate
  min_matches_rank: 0  # Minimum rank for acceptable match
  pix_tol: 2.0  # Pixel tolerance for star matching
  scale_min: 0.95  # Minimum scale factor (when -s flag used)
  scale_max: 1.05  # Maximum scale factor (when -s flag used)
  try_reflection: true  # Try reflected image if normal fails
  print_best_only: false  # Only print successful attempts
  filter_fallback:
    J: ['J', 'H']  # Fallback filters for J-band
    H: ['H', 'J', 'K']  # Fallback filters for H-band
    K: ['K', 'H']  # Fallback filters for K-band
  catalog:
    default_mag: 99.99  # Default magnitude for missing catalog entries
```

#### 3. Photometry Parameters

Controls photometric calibration:

```yaml
photometry:
  fixed_aperture_radius: 10.0  # Fixed aperture for final photometry
  annulus_inner: 15.0  # Inner radius of sky annulus
  annulus_outer: 20.0  # Outer radius of sky annulus
  min_calibration_stars: 3  # Minimum stars for zero-point fit
  sigma_clip: 3.0  # Sigma clipping threshold for outlier rejection
  quality_thresholds:
    rms:
      very_good: 0.05  # RMS < 0.05 mag
      good: 0.10  # RMS < 0.10 mag
      medium: 0.20  # RMS < 0.20 mag
      poor: 0.30  # RMS < 0.30 mag
    rejection:
      good: 0.10  # < 10% rejected
      medium: 0.30  # < 30% rejected
    zp_check:
      very_good: 0.05  # |ZP diff| < 0.05 mag
      good: 0.10  # |ZP diff| < 0.10 mag
      medium: 0.20  # |ZP diff| < 0.20 mag
      poor: 0.30  # |ZP diff| < 0.30 mag
```

#### 4. Image Processing Parameters

```yaml
sky_subtraction:
  edge_fraction: 0.05  # Fraction of image edges to exclude from median

thermal_correction:
  enable: true  # Enable thermal pattern correction
  filter_size: 51  # Median filter size for pattern detection

alignment:
  reference_selection: 'brightest'  # Method: 'brightest' or 'first'
  max_shift: 50  # Maximum allowed shift in pixels

coadd:
  method: 'mean'  # Combination method: 'mean' or 'median'
  sigma_clip: 3.0  # Sigma clipping threshold
  scale_method: 'median'  # Scaling method before combining
```

#### 5. Catalog Parameters

```yaml
catalog:
  service_url: 'http://cdsarc.u-strasbg.fr/viz-bin/asu-tsv'
  catalog_name: '2MASS'
  search_radius: 0.3  # Search radius in degrees
  max_retries: 3  # Maximum download retry attempts
  timeout: 60  # Download timeout in seconds
```

---

## Usage

### Basic Usage

```bash
python remirpipe.py -i /path/to/input/directory
```

This will process all FITS files in the input directory and create output in the same directory.

### Command-Line Options

```
-i, --input DIR          Input directory with FITS files (required)
-o, --output DIR         Output directory (default: same as input)
-c, --config FILE        Configuration file (default: config.yaml in script directory)
-v, --verbose            Enable verbose output
-d, --delete-tmp         Delete tmp directory at the end
-s, --scale-constraint   Apply scale constraints (0.95-1.05) for astrometry
-co, --clean-output      Clean existing output directories before starting
```

### Examples

**Process with custom output directory:**
```bash
python remirpipe.py -i /data/raw -o /data/processed
```

**Verbose mode with custom config:**
```bash
python remirpipe.py -i /data/raw -c my_config.yaml -v
```

**Clean previous results and apply scale constraints:**
```bash
python remirpipe.py -i /data/raw -co -s -v
```

**Process and clean up temporary files:**
```bash
python remirpipe.py -i /data/raw -d
```

---

## Pipeline Workflow

The pipeline processes data through multiple stages:

### Step 1: File Preparation

**Function**: `gunzip_files()`

- Scans input directory recursively for `.fits.gz` files
- Decompresses gzipped files using gzip magic number verification
- Skips macOS resource fork files (`._*`)
- Creates `.fits` files alongside compressed versions

**Function**: `filter_and_prepare_files()`

- Reads FITS headers from all `.fits` files
- Filters out unwanted files:
  - Files without required keywords (FILENAME, IMAGETYP, FILTER, DITHERX, DATE-OBS)
  - Focus frames (IMAGETYP=FOCUS)
  - Files with problematic data (NaN, Inf, zero std deviation)
- Deletes problematic files
- Updates global statistics

### Step 2: File Classification

**Function**: `classify_files()`

- Determines processing type (PROCTYPE) from FITS headers:
  - 0 = FLAT (OBSTYPE=FLATF)
  - 1 = STDSTAR (OBSTYPE=STDSTAR)
  - 2 = SCIENCE (IMAGETYP=OBJECT, not flat/std)
  - -1 = FOCUS (skip)
- Identifies camera system (old/new) from FILENAME pattern
- Copies flats to `reduced/` directory
- Generates preview JPEGs for flats
- Creates calibration master flats (grouped by filter and dither angle)
- Copies science and standard frames to appropriate system directories in `tmp/`

**Grouping Logic**:
- Files grouped by: FILTER, DITHERX, DITHERY, observation sequence
- Groups identified by time gaps (default: 120 seconds between observations)

### Step 3: Sky Subtraction

**Function**: `process_group_sky_subtraction()`

The pipeline creates master sky frames and performs sky subtraction:

1. **Calibration Loading**:
   - Loads flat-field based on filter and dither angle
   - Falls back to median flat if specific flat not found

2. **Frame Processing**:
   - For each frame in group:
     - Apply flat-field correction
     - Normalize flat-fielding
     - Store calibrated frame as `*_c.fits`

3. **Sky Frame Creation**:
   - Median-combines all calibrated frames in group
   - Excludes edge regions (configurable fraction)
   - Applies sigma-clipping to reject outliers
   - Creates master sky: `sky_FILTER_YYYYMMDDThhmmss.fits`

4. **Sky Subtraction**:
   - Subtracts master sky from each calibrated frame
   - Updates headers with processing keywords
   - Adds DATE-OBS averaging if multiple frames combined

**Output**: 
- Calibrated frames: `*_c.fits`
- Master sky: `sky_*.fits` (also copied to `reduced/`)

### Step 4: Thermal Pattern Correction

**Function**: `apply_thermal_correction()`

Removes detector thermal patterns using advanced filtering:

1. **Pattern Detection**:
   - Applies large median filter to each frame (default: 51x51 pixels)
   - Extracts low-frequency thermal pattern

2. **Pattern Removal**:
   - Subtracts detected pattern from frame
   - Preserves astronomical sources and high-frequency structure

3. **Quality Preservation**:
   - Updates FITS headers with thermal correction keywords
   - Maintains flux calibration

**Note**: Can be disabled in config if thermal patterns are negligible.

### Step 5: Frame Alignment

**Function**: `align_frames()`

Aligns frames within each group using cross-correlation:

1. **Reference Selection**:
   - Method 1 (`brightest`): Selects frame with most detected sources
   - Method 2 (`first`): Uses first frame in group
   - Configurable via `alignment.reference_selection`

2. **Shift Calculation**:
   - Uses FFT-based cross-correlation for sub-pixel precision
   - Computes (dx, dy) shift for each frame relative to reference
   - Reports shifts in pipeline log

3. **Frame Registration**:
   - Applies calculated shifts using scipy.ndimage.shift
   - Uses spline interpolation for sub-pixel accuracy
   - Maximum shift constraint prevents bad alignments

4. **Header Updates**:
   - Adds ALIGNED keyword
   - Records shift values (DX_SHIFT, DY_SHIFT)
   - Updates processing date

**Output**: Aligned frames: `*_c_a.fits`

### Step 6: Co-addition

**Function**: `coadd_aligned_frames()`

Combines aligned frames into single co-added image:

1. **Data Preparation**:
   - Loads all aligned frames
   - Optionally scales to common median/mean

2. **Robust Combination**:
   - Method 1 (`mean`): Sigma-clipped mean (default)
   - Method 2 (`median`): Simple median
   - Sigma clipping rejects cosmic rays and outliers

3. **Header Construction**:
   - Averages DATE-OBS from all frames
   - Sums EXPTIME (total exposure)
   - Averages AIRMASS
   - Copies filter, position, and system information
   - Marks incomplete groups with INCOMP keyword

4. **Quality Metrics**:
   - Records number of combined frames (NCOMBINE)
   - Updates global statistics

**Output**: Co-added image: `coadd_FILTER_YYYYMMDDThhmmss.fits`

### Step 7: Catalog Download

**Function**: `download_catalogs_for_groups()`

Downloads 2MASS reference catalogs for astrometry and photometry:

1. **Position Grouping**:
   - Groups co-adds by sky position (RA/DEC)
   - Default grouping radius: configurable in catalog settings

2. **Catalog Query**:
   - Queries INAF/CDS 2MASS catalog service
   - Search radius around target position (default: 0.3 degrees)
   - Downloads J, H, K magnitudes and positions

3. **Caching**:
   - Saves catalogs to `catalogs/` directory
   - Reuses existing catalogs if available
   - Handles download failures with retries

4. **Error Handling**:
   - Logs download failures
   - Pipeline exits if no catalogs downloaded
   - Provides diagnostic messages for troubleshooting

**Output**: Catalog files: `catalog_RA_DEC.csv`

### Step 8: Astrometric Calibration

**Function**: `try_astrometry()`

Performs astrometric calibration using sophisticated geometric quad matching:

#### Algorithm Overview

1. **Multi-Parameter Search**:
   - Tries combinations of detection parameters:
     - `min_pixels`: Minimum connected pixels for sources
     - `threshold_sigma`: Detection threshold
     - Catalog filter: Primary filter + fallbacks (e.g., H → H, J, K)
   - Optional reflection attempt if initial attempts fail

2. **Source Detection**:
   - Uses SEP (Source Extractor Python) for source detection
   - Background estimation and subtraction
   - Connected-component labeling
   - Quality filters (minimum sources, brightness ranking)

3. **Geometric Quad Building**:
   - Creates geometric quads from brightest N sources
   - Quad = 4 stars defining a quadrilateral
   - Computes scale-invariant hash for each quad
   - Uses heap-based selection for efficiency

4. **Quad Matching**:
   - Matches quads between detected sources and catalog
   - Uses geometric similarity (ratio of distances)
   - Finds consistent transformation patterns

5. **Transform Computation**:
   - Computes similarity transform (scale, rotation, translation)
   - Applies transform to all catalog sources
   - Counts matches within pixel tolerance

6. **Transform Validation**:
   - Verifies minimum number of matched stars
   - Checks RMS residual of matched positions
   - Optional scale constraint validation (if `-s` flag used)

7. **WCS Fitting**:
   - Uses matched star pairs to fit WCS
   - astropy.wcs.utils.fit_wcs_from_points
   - Updates FITS header with new WCS solution

#### Attempt Strategy

The pipeline tries multiple approaches in order:

1. **Primary filter detection** (e.g., H-band for H-filter images)
2. **Fallback filters** (e.g., J, K if H fails)
3. **Varied detection parameters** (different thresholds, min_pixels)
4. **Reflection attempt** (try vertically reflected image)

Stops at first successful attempt unless configured otherwise.

#### Success Criteria

An attempt succeeds if:
- Sufficient sources detected (≥ min_sources)
- Quad matches found
- Transform has enough star matches (≥ threshold)
- RMS residual below threshold
- Scale within bounds (if constraint enabled)

**Output**: Updated WCS in co-add and aligned frame headers

### Step 9: Photometric Calibration

**Function**: `photometric_calibration()`

Performs aperture photometry and zero-point calibration:

#### Algorithm Steps

1. **Optimal Aperture Determination**:
   - For each calibration star (stars in catalog):
     - Tries multiple aperture radii
     - Selects aperture with minimum magnitude scatter
   - Uses curve-of-growth analysis

2. **Fixed-Aperture Photometry**:
   - Measures all sources with optimal fixed aperture
   - Applies sky annulus background subtraction
   - Converts counts to instrumental magnitudes

3. **Zero-Point Fitting**:
   - Matches detected sources to catalog stars
   - Computes zero-point for each star: ZP = mag_catalog - mag_instrumental
   - Robust sigma-clipped mean of zero-points
   - Rejects outliers (cosmic rays, bad pixels, etc.)

4. **Calibration Quality Metrics**:
   - RMS of zero-point fit
   - Number of stars used vs. rejected
   - Quality classification (VERY GOOD, GOOD, MEDIUM, POOR)

5. **Full Source Measurement**:
   - Applies zero-point to all detected sources
   - Computes calibrated magnitudes
   - Propagates uncertainties

6. **Catalog Generation**:
   - Creates FITS binary table extension with:
     - X, Y pixel positions
     - RA, DEC (from WCS)
     - Instrumental magnitude
     - Calibrated magnitude
     - Uncertainties
     - Flags (calibration star, edge, saturation)

7. **Calibration Plot**:
   - Plots instrumental vs. catalog magnitudes
   - Shows zero-point fit and residuals
   - Saves as PNG in reduced directory

#### Quality Assessment

**RMS Quality Levels**:
- VERY GOOD: RMS < 0.05 mag
- GOOD: RMS < 0.10 mag
- MEDIUM: RMS < 0.20 mag
- POOR: RMS < 0.30 mag
- VERY POOR: RMS ≥ 0.30 mag

**Rejection Quality**:
- GOOD: < 10% stars rejected
- MEDIUM: 10-30% stars rejected
- POOR: > 30% stars rejected

**Output**: 
- Photometrically calibrated images with catalog extensions
- Calibration plots: `calibration_FILENAME.png`
- Updated FITS headers with zero-point keywords

---

## Output Files

### Directory Structure

```
output_directory/
├── tmp/                          # Temporary processing files
│   ├── old/                      # Old camera system files
│   └── new/                      # New camera system files
├── reduced/                      # Final science-ready products
│   ├── pipelog.txt              # Complete processing log
│   ├── *_flat.fits              # Master flat fields
│   ├── *_flat.jpg               # Flat field previews
│   ├── sky_*.fits               # Master sky frames
│   ├── sky_*.jpg                # Sky frame previews
│   ├── coadd_*.fits             # Co-added science images
│   ├── coadd_*.jpg              # Co-add previews
│   ├── calibration_*.png        # Photometric calibration plots
│   ├── *_STDSTAR_*.fits         # Processed standard star frames
│   └── *_STDSTAR_*.jpg          # Standard star previews
└── catalogs/                     # Downloaded reference catalogs
    └── catalog_*.csv            # 2MASS catalog files
```

### File Naming Convention

- **Calibrated frames**: `original_filename_c.fits`
- **Aligned frames**: `original_filename_c_a.fits`
- **Sky frames**: `sky_FILTER_YYYYMMDDThhmmss.fits`
- **Co-adds**: `coadd_FILTER_YYYYMMDDThhmmss.fits`
- **Master flats**: `FILTER_ditherANGLE_flat.fits`

### FITS Header Keywords Added

#### Processing Keywords
- `PROCTYPE`: Processing type (0=flat, 1=std, 2=sci)
- `FLAT`: Flat field file used
- `SKY`: Sky frame file used
- `SKYSUB`: Sky subtraction applied (T/F)
- `THERMCOR`: Thermal correction applied (T/F)
- `ALIGNED`: Frame alignment applied (T/F)
- `DX_SHIFT`, `DY_SHIFT`: Alignment shifts in pixels
- `COADDED`: Co-addition applied (T/F)
- `NCOMBINE`: Number of frames combined
- `INCOMP`: Incomplete group flag

#### Photometry Keywords
- `ZPPOINT`: Photometric zero-point (mag)
- `ZPPOINT_E`: Zero-point uncertainty (mag)
- `ZPRMS`: RMS of zero-point fit (mag)
- `ZPCALIB`: Number of calibration stars used
- `ZPQUALITY`: RMS quality classification
- `REJQUAL`: Rejection quality classification
- `ZP_CHECK`: Zero-point consistency check
- `MAGLIM`: Limiting magnitude (5-sigma)

### FITS Extensions

Photometrically calibrated images include a binary table extension with:

| Column | Type | Description |
|--------|------|-------------|
| X | float | X pixel coordinate |
| Y | float | Y pixel coordinate |
| RA | float | Right ascension (degrees) |
| DEC | float | Declination (degrees) |
| MAG_INST | float | Instrumental magnitude |
| MAG_INST_ERR | float | Instrumental magnitude error |
| MAG_CAL | float | Calibrated magnitude |
| MAG_CAL_ERR | float | Calibrated magnitude error |
| FLUX | float | Source flux (counts) |
| FLUX_ERR | float | Flux error (counts) |
| IS_CALIB | bool | Calibration star flag |
| FLAGS | int | Quality flags |

---

## Quality Control

### Automated Quality Checks

The pipeline performs comprehensive quality assessment:

#### 1. RMS Quality

Evaluates photometric precision based on zero-point fit RMS:

- Computed from scatter of calibration stars
- Indicates photometric stability
- Threshold-based classification

#### 2. Rejection Quality

Monitors outlier rejection rate:

- High rejection may indicate:
  - Crowded field
  - Poor seeing
  - Detector artifacts
  - Bad WCS solution

#### 3. Zero-Point Consistency

Validates calibration by comparing:

- **Catalog zero-point**: From photometric calibration
- **Standard star zero-point**: From observed standards

Checks include:
- Temporal proximity (standards vs. science)
- Exposure time normalization
- Expected zero-point agreement

### Standard Star Verification

**Function**: `check_standard_stars()`

For STDSTAR observations:
1. Measures instrumental magnitude
2. Compares to catalog magnitude
3. Derives zero-point independently
4. Flags discrepancies with science calibration

### Quality Summary Report

At pipeline completion, comprehensive summary includes:

- Processing statistics (files processed, success/failure rates)
- Astrometry success breakdown by attempt signature
- Photometry results table (zero-point, RMS, limiting magnitude)
- Standard star check results
- Zero-point consistency statistics
- Quality flag distributions
- Total processing time

---

## Advanced Features

### Incomplete Group Handling

The pipeline can process groups with missing frames:

- Identifies groups with < expected number of frames
- Performs processing with available data
- Marks output with `INCOMP` keyword
- Logs incomplete group information

### Reflection Detection

For sources with unusual geometry (e.g., inverted optics):

- Attempts standard astrometry first
- If failed, tries vertically reflected image
- Applies reflection transform to WCS if successful
- Configured via `astrometry.try_reflection`

### Multi-Filter Fallback

If catalog matching fails in image filter:

- Automatically tries related filters
- Example: H-band image can use J or K catalog
- Fallback hierarchy configured per filter
- Maximizes astrometry success rate

### Adaptive Parameter Search

Astrometry uses grid search over parameters:

- Multiple detection thresholds
- Various minimum pixel counts
- Different catalog filters
- Stops at first success (efficient)

### Background Modeling

Sophisticated background estimation:

- Median filtering for coarse background
- Edge exclusion for sky frames
- Annulus-based local background for photometry
- Sigma-clipping for robustness

---

## Troubleshooting

### Common Issues

#### 1. No catalogs downloaded

**Symptoms**: Pipeline exits with "FATAL ERROR: Failed to download any catalogs"

**Causes**:
- No internet connection
- INAF catalog service down
- Firewall blocking requests
- Invalid coordinates in FITS headers

**Solutions**:
- Check internet connectivity
- Verify FITS header RA/DEC values
- Try catalog service URL in browser
- Wait and retry if service is down
- Check firewall settings

#### 2. All astrometry attempts fail

**Symptoms**: "All attempts done for this coadd -> Final: FAILED"

**Causes**:
- Poor WCS initial guess in header
- Too few sources detected
- Catalog doesn't cover field
- Large image distortion

**Solutions**:
- Check FITS header WCS keywords (CRVAL, CRPIX, CD matrix)
- Lower detection threshold in config
- Increase search radius for catalog
- Use `-s` flag to constrain scale
- Verify image quality (focus, seeing)

#### 3. SEP deblending overflow

**Symptoms**: "SEP error (skipping remaining attempts): deblending overflow"

**Causes**:
- Extremely crowded field
- Large connected regions (nebulosity, artifacts)

**Solutions**:
- Increase `threshold_sigma` in config
- Increase `min_pixels` in config
- Pre-process image to remove large-scale structure

#### 4. Poor photometric RMS

**Symptoms**: ZPRMS > 0.2 mag, quality = POOR or VERY POOR

**Causes**:
- Variable seeing
- Crowded field
- Wrong aperture size
- Poor astrometry
- Non-photometric conditions

**Solutions**:
- Check WCS solution quality
- Adjust aperture parameters in config
- Verify catalog match positions
- Inspect calibration plot
- Consider observing conditions

#### 5. High rejection rate

**Symptoms**: REJQUAL = POOR, > 30% stars rejected

**Causes**:
- Crowded field (confusion, blending)
- Artifacts (cosmic rays, bad pixels)
- Poor WCS (wrong star matching)

**Solutions**:
- Inspect image for artifacts
- Check astrometry quality
- Adjust sigma_clip threshold
- Verify catalog match quality

### Debug Strategies

1. **Use verbose mode**: `-v` flag provides detailed logging

2. **Check intermediate files**: Inspect `tmp/` directory for:
   - Calibrated frames (`*_c.fits`)
   - Sky frames quality
   - Aligned frame shifts

3. **Examine headers**: Use `fitsheader` or similar to check:
   - WCS keywords after astrometry
   - Processing keywords
   - Quality metrics

4. **Review log file**: `reduced/pipelog.txt` contains:
   - All processing steps
   - Error messages
   - Attempt details
   - Quality metrics

5. **Inspect plots**: Check calibration PNG files for:
   - Outliers in photometry
   - Systematik trends
   - Zero-point fit quality

### Performance Optimization

**For large datasets**:

- Use `-d` flag to delete temporary files
- Use `-co` flag to clean previous runs
- Process in smaller batches
- Monitor disk space

**For slow astrometry**:

- Reduce `num_quads` in config
- Limit `top_matches` to evaluate
- Enable `print_best_only` to reduce logging
- Use fewer parameter combinations

---

## Pipeline Summary

The REMIR pipeline provides a complete, automated solution for processing near-infrared astronomical images. Key strengths include:

- **Robustness**: Multiple fallback strategies for astrometry
- **Automation**: Minimal user intervention required
- **Quality**: Comprehensive metrics and validation
- **Flexibility**: Highly configurable via YAML
- **Transparency**: Detailed logging and intermediate products

The pipeline transforms raw telescope data into science-ready, calibrated images with accurate astrometry and photometry, suitable for time-domain astronomy, photometric studies, and archival analysis.

---

## Additional Resources

### Related Tools

- **DS9**: FITS image visualization
- **Topcat**: Catalog analysis and visualization
- **Astropy**: Python astronomy library
- **STILTS**: Catalog manipulation

### References

- **2MASS Catalog**: [IPAC](https://www.ipac.caltech.edu/2mass/)
- **SEP**: [Source Extractor Python](https://sep.readthedocs.io/)
- **Astropy WCS**: [Documentation](https://docs.astropy.org/en/stable/wcs/)
- **REM Telescope**: [INAF](http://www.rem.inaf.it/)

---

## Contact & Support

For issues specific to this pipeline implementation, check:

1. Log files in `reduced/pipelog.txt`
2. FITS headers for processing keywords
3. Calibration plots for photometry issues
4. Configuration file settings

---

**Version**: 1.0  
**Last Updated**: February 2026  
**Author**: REMIR Pipeline Development Team

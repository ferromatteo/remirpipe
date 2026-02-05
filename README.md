# REMIR Pipeline

Complete automated reduction and analysis pipeline for REMIR (REM InfraRed) FITS data.

## Overview

The REMIR pipeline performs end-to-end processing of near-infrared imaging data, from raw FITS files through calibration, alignment, co-addition, astrometric calibration, and photometric calibration. The pipeline is designed to handle dithered observations in J, H, and K bands with automatic source detection, catalog matching, and quality assessment.

## Features

- **Automated calibration**: Bad pixel masking, flat field correction, thermal pattern subtraction
- **Sky subtraction**: Sigma-clipped median sky estimation from dithered frames
- **Frame alignment**: Geometric dither positioning with optional cross-match refinement
- **Co-addition**: Noise-weighted combination of aligned frames
- **Astrometric calibration**: Quad-matching algorithm with 2MASS/VSX catalog
- **Photometric calibration**: Automated zeropoint fitting with outlier rejection
- **Quality assessment**: RMS metrics, rejection statistics, standard star checks
- **Batch processing**: Handles multiple targets, filters, and observation blocks

## Requirements

### Python Dependencies

```bash
pip install numpy scipy astropy photutils matplotlib pyyaml
```

### Required packages:
- `numpy` - Array operations
- `scipy` - Image processing, spatial operations
- `astropy` - FITS I/O, WCS, coordinate transformations
- `photutils` - Aperture photometry
- `matplotlib` - Preview generation
- `pyyaml` - Configuration file parsing

### Calibration Files

Place calibration files in the `data_folder` (configured in `config.yaml`):

```
data_2026_01/
├── pixel_mask.fits                          # Bad pixel mask (0=bad, 1=good)
├── J_dither0_flat.fits                      # Flat fields per filter/dither
├── J_dither72_flat.fits
├── H_dither0_flat.fits
├── ...
├── J_dither0_exptime30_thermal.fits         # Thermal templates (optional)
└── ...
```

## Installation

1. Clone or download the pipeline files:
```bash
cd /path/to/your/workspace
```

2. Ensure `remirpipe.py` and `config.yaml` are in the same directory

3. Make the script executable (optional):
```bash
chmod +x remirpipe.py
```

4. Install Python dependencies (see Requirements above)

## Usage

### Basic Usage

```bash
python remirpipe.py -i /path/to/input/data -o /path/to/output -v
```

### Command-Line Options

```
-i, --input DIR          Input directory with FITS files (required)
-o, --output DIR         Output directory (default: same as input)
-c, --config FILE        Configuration file (default: config.yaml in script directory)
-v, --verbose            Verbose output to console
-d, --delete-tmp         Delete tmp/ directory after completion
-s, --scale-constraint   Apply scale constraints (0.95-1.05) for astrometry
-co, --clean-output      Clean existing output directories before starting
```

### Examples

**Process data with verbose output:**
```bash
python remirpipe.py -i ./observations/2026-01-15 -v
```

**Process with custom config and clean output:**
```bash
python remirpipe.py -i ./data -o ./reduced -c custom_config.yaml -co -v
```

**Apply strict astrometry and delete temp files:**
```bash
python remirpipe.py -i ./data -s -d -v
```

## Pipeline Workflow

The pipeline executes the following steps automatically:

### 1. **File Preparation**
- Decompress `.fits.gz` files
- Filter files by DITHID (removes incomplete sequences)
- Fix invalid header values
- Add FILENAME and PROCTYPE keywords
- Apply bad pixel mask (sets masked pixels to NaN)

### 2. **File Classification**
- Classify by system (old/new dither mechanism)
- Separate FLAT, FOCUS, STD, and SCI observations
- Move files to appropriate directories

### 3. **Grouping & Sky Subtraction**
- Group by OBJECT/FILTER/OBSID/SUBID with time constraints
- Validate dither completeness (5 positions expected)
- Apply flat field correction per filter/dither
- Compute sigma-clipped median sky from dithered frames
- Subtract sky pattern
- Apply thermal pattern correction (static or dynamic templates)

### 4. **Alignment & Co-addition**
- Compute geometric shifts from dither pattern
- Optional: Refine alignment via source cross-matching
- Shift frames to common grid with subpixel accuracy
- Co-add with noise-weighted combination
- Generate incomplete co-adds when needed

### 5. **Catalog Download**
- Query 2MASS catalog for field sources (J, H, K photometry)
- Query VSX for known variable stars
- Cache catalogs per sky position

### 6. **Astrometric Calibration**
- Detect sources with SEP (configurable thresholds)
- Build geometric quads from brightest sources
- Match detection/catalog quads via geometric hashing
- Fit WCS with SIP distortion correction
- Validate solution (match count, RMS residual)
- Try multiple detection parameters until success

### 7. **Photometric Calibration**
- Detect sources in calibrated images
- Filter isolated, central sources
- Match detections to 2MASS catalog
- Fit zeropoint with iterative sigma clipping
- Calculate limiting magnitude
- Generate diagnostic plots
- Quality assessment (RMS, rejection rate)

### 8. **Standard Star Validation**
- Check PROCTYPE=1 observations against catalog
- Compare field vs standard star zeropoints
- Flag inconsistencies

## Output Structure

```
output/
├── tmp/                                      # Intermediate files
│   ├── old/                                  # Pre-2025 dither system
│   │   └── [grouped files]
│   ├── new/                                  # Post-2025 dither system
│   │   └── [grouped files]
│   ├── skysub/                              # Sky-subtracted frames
│   │   ├── OBJECT_OBSID_SUBID_FILTER_sky.fits
│   │   └── [individual sky-subtracted files]
│   ├── aligned/                             # Aligned frames
│   │   └── [shifted frames ready for co-add]
│   └── coadd/                               # Co-added images
│       └── OBJECT_OBSID_SUBID_FILTER_coadd.fits
├── catalogs/                                # Downloaded 2MASS/VSX catalogs
│   └── catalog_RA_DEC.fits
├── reduced/                                 # Final calibrated products
│   ├── [coadd files with WCS]
│   ├── [photometry catalogs .cat]
│   ├── [photometry plots .png]
│   └── [preview images .jpg] (if enabled)
└── pipelog.txt                              # Complete processing log
```

### Output File Types

- **Sky frames** (`*_sky.fits`): DITHID=98, median sky pattern
- **Co-adds** (`*_coadd.fits`): DITHID=99, combined aligned frames
- **Reduced** (`reduced/*.fits`): Final images with WCS headers
- **Catalogs** (`*.cat`): Photometry results with calibrated magnitudes
- **Plots** (`*.png`): Instrumental vs catalog magnitude diagnostics
- **Previews** (`*.jpg`): Quick-look images (optional)

## Configuration

All pipeline parameters are controlled via `config.yaml`. Key sections:

### Paths
- `data_folder`: Location of calibration files

### Calibration Toggles
```yaml
calibration:
  enable_pixel_mask: true      # Bad pixel correction
  enable_flat_correction: true # Flat fielding
  enable_thermal_correction: true  # Thermal pattern subtraction
  thermal_use_static: true     # Use pre-computed vs dynamic templates
```

### Detection Parameters
```yaml
detection:
  min_pixels: [10, 5]          # Connected pixel threshold (try in order)
  threshold_sigma: [2.0, 1.2]  # Detection sigma (try in order)
  aperture_radius: 3.5         # Photometry aperture [pixels]
```

### Astrometry
```yaml
astrometry:
  min_matches_rank: 3          # Min stars for valid transform
  min_match_fraction: 0.12     # Min fraction of sources matched
  accept_rms_px: 1.5           # Max RMS residual [pixels]
  filter_fallback:
    J: ["J", "H"]              # Try J mags, then H
    K: ["K", "H"]
    H: ["H"]
```

### Photometry
```yaml
photometry:
  enabled: true
  aperture_radius: 3.0         # Fixed aperture [pixels]
  central_fraction: 0.90       # Use central 90% of image
  min_isolation_dist: 8.0      # Min source separation [pixels]
  sigma_clip: 3.0              # Outlier rejection threshold
  min_calibration_stars: 3     # Min stars for valid calibration
```

See `config.yaml` for complete documentation of all parameters.

## Quality Metrics

The pipeline reports several quality indicators:

### RMS Quality (photometric fit)
- **VERY GOOD**: RMS < 0.1 mag
- **GOOD**: RMS < 0.175 mag
- **MEDIUM**: RMS < 0.25 mag
- **POOR**: RMS < 0.35 mag
- **VERY POOR**: RMS ≥ 0.35 mag

### Rejection Quality
- **GOOD**: < 25% stars rejected
- **MEDIUM**: 25-50% rejected
- **POOR**: ≥ 50% rejected

### Zeropoint Comparison (Standard vs Field)
- **VERY GOOD**: |ΔZP| < 0.05 mag
- **GOOD**: |ΔZP| < 0.1 mag
- **MEDIUM**: |ΔZP| < 0.2 mag
- **POOR**: |ΔZP| ≥ 0.2 mag

## FITS Header Keywords

The pipeline adds/modifies the following keywords:

- `PROCTYPE`: Processing type (0=FLAT, 1=STD, 2=SCI, -1=FOCUS)
- `PROCSTAT`: Processing status (1=reduced)
- `PSTATSUB`: Processing sub-status (1=sky, 2=skysub, 3=aligned, 4=coadd)
- `DITHID`: Dither position (0-4) or special (98=sky, 99=coadd)
- `FILENAME`: Original filename
- `NCOADD`: Number of frames co-added
- `ZP_*`: Photometric zeropoint and quality metrics
- `MAG_LIM`: 3σ limiting magnitude
- `HISTORY`: Processing steps and parameters

## Troubleshooting

### Common Issues

**No sources detected:**
- Lower `threshold_sigma` in config (try 1.0-1.5)
- Reduce `min_pixels` (try 3-5)
- Check if image is properly sky-subtracted

**Astrometry fails:**
- Enable `scale_constraint: false` (try without -s flag)
- Check initial WCS in header (RA, DEC, CRPIX)
- Increase `catalog_radius_arcmin` for wider search
- Try `try_reflection: true` if image may be flipped

**Photometry calibration fails:**
- Check filter is in `calibrate_filters` list (J/H/K only)
- Verify 2MASS catalog has sufficient stars
- Lower `min_calibration_stars` if field is sparse
- Increase `match_tolerance` for low S/N images

**Flat field not found:**
- Check filename format: `{FILTER}_dither{ANGLE}_flat.fits`
- Verify `data_folder` path in config
- Dither angles should be 0, 72, 144, 216, 288

### Debug Tips

1. Run with `-v` flag for detailed output
2. Check `pipelog.txt` for complete processing log
3. Examine intermediate files in `tmp/` directory
4. Use `-co` flag to start fresh (removes old outputs)
5. Test with single observation before batch processing

## Performance

Typical processing times (approximate):
- File preparation: ~1-2 sec per file
- Sky subtraction: ~2-5 sec per group
- Alignment: ~1-2 sec per co-add
- Astrometry: ~5-15 sec per co-add (depends on attempts)
- Photometry: ~2-5 sec per co-add

Total: ~30-60 seconds per observation block (varies with field complexity)

## Pipeline Statistics

The pipeline tracks and reports:
- Files processed by type (FLAT/STD/SCI/FOCUS)
- Groups (complete/incomplete/defective)
- Sky frames created
- Co-adds generated
- Astrometry success/failure rate
- Photometry calibrations performed

Statistics are printed at pipeline completion.

## Authors & Credits

REMIR Pipeline developed for the REM telescope data reduction.

**Dependencies:**
- [Astropy](https://www.astropy.org/) - Astronomy Python library
- [Photutils](https://photutils.readthedocs.io/) - Photometry tools
- [SEP](https://sep.readthedocs.io/) - Source extraction (via photutils)
- [2MASS Point Source Catalog](https://irsa.ipac.caltech.edu/Missions/2mass.html) - Astrometric/photometric reference
- [VSX](https://www.aavso.org/vsx/) - Variable Star Index

## License

See project repository for license information.

## Version History

- **v1.0** (2026-01) - Initial release with complete reduction pipeline
  - Bad pixel masking, flat correction, thermal correction
  - Automated astrometry via quad matching
  - Photometric calibration with 2MASS
  - Quality assessment and standard star validation

---

For detailed parameter descriptions, see comments in `config.yaml`.  
For algorithm details, review inline documentation in `remirpipe.py`.

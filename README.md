# Acoustic Differential-Area Fish Counting

Reproducibility materials for the acoustic differential-area method used to
estimate fish quantity in large offshore cages from omnidirectional-sonar
images.

The field campaign was conducted by the same research team as Sun et al.
(2025) and shares the Dehai No. 1 platform, *Trachinotus ovatus* stock,
FishScan II sonar, two cages, and survey campaign. This repository documents a
different processing method: distance-aware annular segmentation, single-fish
pixel calibration, and Bayesian-regularised BP correction.

## Method summary

- Images are resized to 800 x 600 pixels.
- Fifteen frames are selected from each scan dataset.
- Each frame is divided into 59 concentric annular zones of 5 pixels.
- Otsu thresholding is applied independently within each ring.
- Foreground pixels are converted using the distance-specific single-fish
  calibration for that ring.
- The 59 calibrated ring counts are summed within each frame, and the 15
  selected water-layer totals are summed to one scalar direct estimate.
- The scalar direct estimate is supplied to a one-hidden-layer BP network with
  30 neurons and one fish-quantity output.
- MATLAB Bayesian regularisation (`trainbr`) is used for model fitting.
- Each cage has 27 datasets: 18 for model development and 9 for independent
  validation. Coordinate observations within a scan are nested measurements,
  not independent cage-level replicates.
- Dataset 24 must be marked invalid and excluded from revised validation.

See [ALGORITHM.md](ALGORITHM.md) for the complete pseudocode.

## Repository structure

```text
src/extract_ring_features.py   Ring-wise image processing and calibration
matlab/train_bp_model.m        BP training and independent validation
matlab/loocv_bp.m              Leave-one-dataset-out validation
examples/calibration_template.csv
examples/dataset_features_template.csv
requirements.txt
```

## Python preprocessing

```powershell
python -m pip install -r requirements.txt
python src/extract_ring_features.py sonar_frames calibration.csv output_dir
```

Optional arguments support a binary cage mask and a configurable frame-level
aggregation rule. Record the selected rule when reproducing a reported result.

## MATLAB fitting

The input table must follow `examples/dataset_features_template.csv` and contain
`ring_01` through `ring_59`, `reference_count`, and `valid`.

```matlab
results = train_bp_model("dataset_features.csv", "Cage1");
cv = loocv_bp("dataset_features.csv");
```

MATLAB requires Deep Learning Toolbox (formerly Neural Network Toolbox). The
exact MATLAB release used in the original analysis was not retained in the
archived computational record. The released Python reference implementation is
tested with Python 3.10 or later and the minimum package versions in
`requirements.txt`.

## Data availability

Raw sonar images and commercial farm-operation records are not included because
they contain facility-specific operational information. The repository provides
the full algorithms, parameter settings, input templates, and the derived
59-ring single-fish calibration vector. Anonymised
intermediate data may be made available by the corresponding author subject to
the study's data-sharing conditions.

## Reference

Sun, P., Huang, X., Sun, J., Tao, Q., Yuan, T., Li, G., Pang, G., Liu, H., and
Hu, Y. (2025). Estimating fish quantity and distribution in offshore cage
aquaculture using YOLOv8 and fish density. *Smart Agricultural Technology*, 12,
101514. https://doi.org/10.1016/j.atech.2025.101514

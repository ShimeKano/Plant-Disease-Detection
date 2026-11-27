# Dataset inventory

This document lists datasets that can be verified from the current repository's training code/notebooks, archived training metadata, and project history.

## Confirmed dataset

### PlantVillage — TensorFlow Datasets

- TFDS builder: `plant_village`
- TFDS version: `1.0.2` was observed in an earlier cache inspection. The current notebook does **not** explicitly pin the version, so future runs should record the resolved version in their run metadata.
- Source: https://www.tensorflow.org/datasets/catalog/plant_village
- Role: training and held-out evaluation for the 15-class MobileNetV2 pipeline.
- Download: automatic through `tfds.builder("plant_village")` and `download_and_prepare()`.
- Colab cache: `/content/tfds_cache`
- Filtered image folders: `/content/plantvillage_15`
- Archived run: 22,785 images across 15 classes.
- Archived split: 15,949 train / 3,418 validation / 3,418 test (70/15/15, stratified, seed 42).

See [dataset_catalog.json](dataset_catalog.json) for class-by-class counts and machine-readable details.

## Other datasets

No other training dataset is currently confirmed by the repository or project notes reviewed. In particular, images uploaded later for manual testing must be recorded separately as an **external test set**, including collection date, location/context, labels, consent/usage rights, and whether images overlap with training data.

## Important reproducibility note

The class counts and split sizes above describe the archived training run. They are not guaranteed to match every future TFDS download. Every new training run should save:
- TFDS builder name and resolved version;
- download/source date and dataset fingerprint/checksum where available;
- class counts after filtering;
- split seed and exact sample identifiers per split;
- preprocessing and augmentation configuration;
- model run ID and resulting metrics.

PlantVillage is not a substitute for field validation. Keep field images separate from training/validation/test data unless a documented new experiment intentionally changes the split.

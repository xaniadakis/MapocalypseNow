# Mapocalypse Now – Land Cover Classification with U-Net

## Overview
This repository contains a complete pipeline for land cover classification using Sentinel-2 imagery. 
It includes preprocessing, pan-sharpening, CRS alignment, dataset preparation, model training, evaluation and prediction on an unseen region.

## Structure
- `data/`: Raw and processed data
- `assets/`: Training metrics and other visualizations
- `src/`: Source code for preprocessing and model training
- `checkpoints/`: Saved model weights and information
- `venv/`: Python virtual environment

## Setup
- Create a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
- Preprocess the data:
    ```bash
    python src/i_pansharpening.py
    python src/ii_crs_alignment.py
    python src/iii_merge.py
    python src/iv_build_images.py
    ```

- Prepare training data:
    ```bash
    python src/v_prepare_training_data.py
    ```

- Train the model:
    ```bash
    python src/vii_train_unet.py
    ```

- Predict on unseen region:
    ```bash
    python src/viii_predict_unseen_region.py
    ```


## Notes
- Uses 13-band Sentinel-2 input and an enhanced U-Net with ResNet encoder.
- Augmentations (spatial & radiometric) are applied only on the training set.
- Training/validation/test splits are automatically created and saved under `data/patch_dataset_<PATCH_SIZE>_<STRIDE>/splits/`, where `<PATCH_SIZE>` and `<STRIDE>` refer to the dimensions and stride used when extracting image patches.
- Sentinel-2 tiles (for both training and inference) are expected in the `data/` directory.
- Ground truth raster masks for training must also be placed in `data/`.
- Ground truth annotations for the unseen region used in inference evaluation should be placed in `data/` as well.

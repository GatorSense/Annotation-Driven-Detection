# Annotation-Driven-Detection
Beyond Task-Driven Features for Object Detection

Managed by [Meilun Zhou](https://www.linkedin.com/in/meilun-zhou/) through [GatorSense](https://faculty.eng.ufl.edu/machine-learning/).

## Environment Setup

This project requires Python 3.8+ and the following packages:

- numpy>=1.21.0
- pandas>=1.3.0
- scikit-learn>=1.0.0
- tensorflow>=2.8.0
- keras>=2.8.0
- comet-ml>=3.30.0
- matplotlib>=3.4.0
- seaborn>=0.11.0
- umap-learn>=0.5.3
- torch>=1.10.0
- torchvision>=0.11.0
- pillow

To install all dependencies:

```bash
pip install -r requirements.txt
pip install torch torchvision pillow
```

## Overview

This repository provides the full experimental pipeline for evaluating latent space methods (DTL, DTL-hard, MATL, CTL) for object detection on the AWIR dataset using Faster R-CNN. The code is organized into modular scripts and utilities for training, feature extraction, and evaluation.

### Main Python Scripts

- **embedding_model_training.py**: Train projection heads from CLIP embeddings using various triplet loss formulations.
- **frcnn_embedding_training.py**: Train Faster R-CNN models with latent grid fusion using precomputed sliding-window latent feature grids.
- **awir_custom_losses.py**: Custom loss functions (e.g., triplet losses) for embedding training.
- **awir_utilities.py**: Utility functions for feature computation, normalization, and data processing.

### Pipeline Stages

1. **Train Projection Models from CLIP Embeddings**
    - Script: `embedding_model_training.py`
    - Purpose: Train a projection head that maps frozen CLIP ViT-B/32 image embeddings into a structured latent space (DTL, DTL_HARD, MATL, CTL) using triplet loss variants.
    - Output: Trained projection head checkpoints (e.g., `ctl_best_fold1.h5`, `dtl_best_fold1.h5`, etc.) saved in `demo_trained_embedding_models/`.

2. **Generate Sliding-Window Latent Feature Grids**
    - Notebook: `observing_embeddings.ipynb`
    - Purpose: For each projection model, slide a window across each AWIR image, extract patch-level CLIP embeddings, project them, and save as spatial grids in `.npz` format under `demo_sliding_grids/`.

3. **Train Faster R-CNN with Latent Grid Fusion**
    - Script: `frcnn_embedding_training.py`
    - Purpose: Train Faster R-CNN using precomputed latent grids, evaluating different fusion techniques (additive, concatenation, FiLM, spatial mask, etc.).
    - Output: Trained detection models and logs.

4. **Evaluate Detection Models**
    - Notebook: (e.g., `faster_r_cnn.ipynb` or custom evaluation code)
    - Purpose: Load trained models, run inference on validation data, and compute metrics (mAP, precision, recall, etc.).

---

## Usage

### 1. Train Embedding Projection Models

```bash
python embedding_model_training.py --margin 0.1 --embedding_dimension 1024 --batch_size 32 --modality rgb --test_size 0.5 --cls_weight 0.5
```
*See script arguments in the file for more options.*

### 2. Generate Sliding-Window Grids

Run the relevant section in `observing_embeddings.ipynb` to save embedding grids for each image. Output is saved in `demo_sliding_grids/`.

### 3. Train Faster R-CNN with Grid Fusion

```bash
python frcnn_embedding_training.py --data_root <AWIR dataset root> --grid_root demo_sliding_grids/ --variants dtl,dtl_hard,matl,ctl --epochs 20 --batch 2 --num_workers 4 --run_dir <output_dir> --run_tag <experiment_tag>
```
*See script and docstring for all available arguments.*

### 4. Evaluate Detection Models

Use your preferred evaluation notebook or script to load trained models and compute detection metrics.

---

## Demo: Using Precomputed Grids with Faster R-CNN

This demo shows how to use the precomputed sliding-window latent feature grids in `demo_sliding_grids/` for grid-fused Faster R-CNN training.

### 1. Prepare the Grids

The folder `demo_sliding_grids/` contains `.npz` files for each image, with keys like `dtl_grid`, `dtl_hard_grid`, `matl_grid`, `ctl_grid`, etc. These are spatial grids of projected CLIP features, precomputed for each image in the dataset.

Example structure:

```
demo_sliding_grids/
    train/
        Cow__CA_DJI_0488.npz
        ...
    val/
        ...
    index_train.csv
    index_val.csv
```

### 2. Train Faster R-CNN with Grid Fusion

Run the following command to train a grid-fused Faster R-CNN using the precomputed grids:

```bash
python frcnn_embedding_training.py \
    --data_root <AWIR dataset root> \
    --grid_root demo_sliding_grids \
    --variants dtl,dtl_hard,matl,ctl \
    --epochs 20 \
    --batch 2 \
    --num_workers 4 \
    --run_dir <output_dir> \
    --run_tag demo_grid_fusion \
    --fusion additive
```

- `--grid_root` should point to the folder containing the precomputed `.npz` grids (e.g., `demo_sliding_grids`).
- `--variants` specifies which grid types to use (must match keys in the `.npz` files).
- `--fusion` selects the fusion method (additive, residual_concat, film, spatial_mask).

The script will automatically load the correct grid for each image and fuse it into the FPN backbone during training.

### 3. Output

Checkpoints and logs will be saved in the specified `--run_dir` under the `--run_tag` subfolder. Each variant will have its own best checkpoint based on validation loss.

---

## File Descriptions

- **embedding_model_training.py**: Main script for training embedding projection heads. Supports argument parsing for margin, embedding dimension, batch size, modality, and more. Uses utilities and custom losses from other modules.
- **frcnn_embedding_training.py**: Main script for training Faster R-CNN with latent grid fusion. Supports various fusion strategies and experiment tracking.
- **awir_custom_losses.py**: Implements TensorFlow/Keras triplet loss functions and related selection logic.
- **awir_utilities.py**: Provides feature computation, normalization, and data processing utilities (e.g., mutual information, ANOVA, grid generation).

---

## Pipeline Summary

1. Train projection heads from CLIP embeddings (`embedding_model_training.py`).
2. Generate sliding-window latent feature grids per fold (`observing_embeddings.ipynb`).
3. Train Faster R-CNN using grid fusion (`frcnn_embedding_training.py`).
4. Evaluate trained detection models (custom evaluation code or notebook).

Data flow:

AWIR Images → CLIP Embeddings → Projection Head (DTL / MATL / CTL) → Sliding Window Grid → Faster R-CNN Fusion → Detection Metrics

